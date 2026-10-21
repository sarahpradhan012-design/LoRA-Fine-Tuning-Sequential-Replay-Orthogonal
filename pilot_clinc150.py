"""Two-domain GPU memory pilot for sequential LoRA on CLINC150."""

from __future__ import annotations

import gc
import json
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

from experiment_io import (
    TASK_ORDER,
    accepted_prediction,
    available_intents,
    format_prompt,
    format_target,
    is_correct_prediction,
    normalize_generated_output,
)


ROOT = Path(__file__).resolve().parent
TRAIN_FILE = ROOT / "data" / "prepared" / "train.jsonl"
VALIDATION_FILE = ROOT / "data" / "prepared" / "validation.jsonl"
RUN_ROOT = ROOT / "run-logs" / "pilot"
ADAPTER_ROOT = ROOT / "artifacts" / "pilot"
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
SEED = 42
DOMAINS = TASK_ORDER[:2]
BASE_MAX_LENGTH = 128
OOM_RETRY_LENGTH = 96
LORA_RANK = 4
LORA_ALPHA = 8
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "v_proj"]
PER_DEVICE_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
EPOCHS_PER_DOMAIN = 1
LEARNING_RATE = 1e-4
MAX_GRAD_NORM = 1.0
MAX_NEW_TOKENS = 16


class PilotOOM(RuntimeError):
    """Signals an OOM that needs a whole-pilot retry or CPU fallback."""


class PromptTargetDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], tokenizer, max_length: int):
        self.items: list[dict[str, list[int]]] = []
        for row in rows:
            prompt_content = format_prompt(row["text"])
            user_turn = [{"role": "user", "content": prompt_content}]
            prompt_ids = tokenizer.apply_chat_template(
                user_turn, tokenize=True, add_generation_prompt=True
            )
            full_ids = tokenizer.apply_chat_template(
                user_turn + [{"role": "assistant", "content": format_target(row["intent"])}],
                tokenize=True,
                add_generation_prompt=False,
            )
            if full_ids[:len(prompt_ids)] != prompt_ids:
                raise ValueError("Qwen chat-template training prefix is not a full-sequence prefix")
            if len(full_ids) > max_length:
                raise ValueError(
                    f"Formatted example has {len(full_ids)} tokens, exceeding max length {max_length}; "
                    "the intent target will not be truncated silently."
                )
            labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
            self.items.append({"input_ids": full_ids, "labels": labels})

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.items[index]


class DynamicPadCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        width = max(len(item["input_ids"]) for item in features)
        input_ids, attention_mask, labels = [], [], []
        for item in features:
            padding = width - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [self.pad_token_id] * padding)
            attention_mask.append([1] * len(item["input_ids"]) + [0] * padding)
            labels.append(item["labels"] + [-100] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def read_jsonl(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def capture_nvidia_smi(path: Path) -> None:
    result = subprocess.run(["nvidia-smi"], check=False, capture_output=True, text=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result.stdout or result.stderr, encoding="utf-8")


def is_cuda_oom(error: BaseException) -> bool:
    message = str(error).lower()
    return isinstance(error, torch.cuda.OutOfMemoryError) or (
        isinstance(error, RuntimeError) and "out of memory" in message and "cuda" in message
    )


def snapshot_adapter(model) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in get_peft_model_state_dict(model).items()
    }


def restore_adapter(model, state: dict[str, torch.Tensor]) -> None:
    set_peft_model_state_dict(model, state)


def build_training_arguments(domain: str, max_length: int, device: str) -> TrainingArguments:
    work_dir = RUN_ROOT / "work" / f"{domain}-seq{max_length}-{device}"
    return TrainingArguments(
        output_dir=str(work_dir),
        num_train_epochs=EPOCHS_PER_DOMAIN,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="constant",
        warmup_steps=0,
        weight_decay=0.0,
        optim="adamw_torch",
        max_grad_norm=MAX_GRAD_NORM,
        fp16=(device == "cuda"),
        use_cpu=(device == "cpu"),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_strategy="steps",
        logging_steps=25,
        save_strategy="no",
        eval_strategy="no",
        report_to=[],
        seed=SEED,
        data_seed=SEED,
        dataloader_num_workers=0,
        remove_unused_columns=True,
        disable_tqdm=False,
    )


def create_lora_model(revision: str, dtype: torch.dtype, device: str):
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=revision,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    base_model.config.use_cache = False
    base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    base_model.enable_input_require_grads()
    model = get_peft_model(base_model, LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    ))
    if device == "cuda":
        model.to("cuda")
    model.config.use_cache = False
    unexpected_trainables = [
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad and "lora_" not in name
    ]
    if unexpected_trainables:
        raise RuntimeError(f"Non-LoRA parameters are trainable: {unexpected_trainables[:10]}")
    return model


def train_domain(
    model,
    tokenizer,
    rows: list[dict[str, str]],
    domain: str,
    max_length: int,
    device: str,
    start_state: dict[str, torch.Tensor],
    attempt_dir: Path,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    restore_adapter(model, start_state)
    train_rows = [row for row in rows if row["domain"] == domain]
    if len(train_rows) != 1500:
        raise ValueError(f"Expected 1,500 training rows for {domain}, found {len(train_rows)}")
    dataset = PromptTargetDataset(train_rows, tokenizer, max_length)
    arguments = build_training_arguments(domain, max_length, device)
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=dataset,
        data_collator=DynamicPadCollator(tokenizer.pad_token_id),
    )

    attempt_dir.mkdir(parents=True, exist_ok=True)
    capture_nvidia_smi(attempt_dir / "nvidia-smi-before.txt")
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    train_error: BaseException | None = None
    train_result = None
    try:
        train_result = trainer.train()
        if device == "cuda":
            torch.cuda.synchronize()
    except BaseException as error:
        if not is_cuda_oom(error):
            raise
        train_error = error
    duration = time.perf_counter() - started
    peak_allocated = torch.cuda.max_memory_allocated() if device == "cuda" else None
    peak_reserved = torch.cuda.max_memory_reserved() if device == "cuda" else None
    capture_nvidia_smi(attempt_dir / "nvidia-smi-after.txt")
    steps = int(trainer.state.global_step)
    loss = None if train_result is None else float(train_result.training_loss)
    del trainer, dataset
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    details: dict[str, Any] = {
        "domain": domain,
        "device": device,
        "max_sequence_length": max_length,
        "examples": len(train_rows),
        "training_seconds": duration,
        "optimizer_updates": steps,
        "training_loss": loss,
        "peak_vram_allocated_gib": None if peak_allocated is None else peak_allocated / 1024**3,
        "peak_vram_reserved_gib": None if peak_reserved is None else peak_reserved / 1024**3,
        "oom": train_error is not None,
        "oom_message": None if train_error is None else str(train_error)[:1000],
    }
    if train_error is not None:
        raise PilotOOM(json.dumps(details))
    return details, snapshot_adapter(model)


def score_stage(
    model,
    tokenizer,
    validation_rows: list[dict[str, str]],
    task_index: int,
    device: str,
    max_length: int,
    output_path: Path,
) -> dict[str, Any]:
    seen = available_intents(task_index)
    eval_rows = [row for row in validation_rows if row["domain"] in set(DOMAINS[:task_index])]
    correct = invalid = 0
    by_domain: dict[str, dict[str, int]] = {
        domain: {"correct": 0, "total": 0, "invalid": 0}
        for domain in DOMAINS[:task_index]
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    with output_path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in eval_rows:
            prompt_content = format_prompt(row["text"])
            prompt_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt_content}],
                tokenize=True,
                add_generation_prompt=True,
            )
            if len(prompt_ids) > max_length:
                raise ValueError(
                    f"Evaluation prompt has {len(prompt_ids)} tokens, exceeding {max_length}"
                )
            input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            attention_mask = torch.ones_like(input_ids)
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    num_beams=1,
                    temperature=1.0,
                    top_p=1.0,
                    top_k=50,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )
            raw = tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True)
            accepted = accepted_prediction(raw, task_index)
            is_correct = is_correct_prediction(raw, row["intent"], task_index)
            invalid += int(accepted is None)
            correct += int(is_correct)
            by_domain[row["domain"]]["total"] += 1
            by_domain[row["domain"]]["correct"] += int(is_correct)
            by_domain[row["domain"]]["invalid"] += int(accepted is None)
            stream.write(json.dumps({
                "task_index": task_index,
                "available_intent_count": len(seen),
                "domain": row["domain"],
                "text": row["text"],
                "target": row["intent"],
                "raw_output": raw,
                "normalized_output": normalize_generated_output(raw),
                "accepted_prediction": accepted,
                "correct": is_correct,
            }, ensure_ascii=False) + "\n")
    model.train()
    for stats in by_domain.values():
        total = stats["total"]
        stats["accuracy"] = stats["correct"] / total if total else 0.0
    return {
        "task_index": task_index,
        "seen_domains": list(DOMAINS[:task_index]),
        "available_intents": len(seen),
        "validation_examples": len(eval_rows),
        "correct": correct,
        "accuracy": correct / len(eval_rows) if eval_rows else 0.0,
        "invalid_outputs": invalid,
        "per_domain": by_domain,
        "prediction_file": str(output_path.relative_to(ROOT)),
    }


def run_two_domain_pilot(model, tokenizer, train_rows, validation_rows, revision: str,
                         device: str, max_length: int, run_label: str) -> dict[str, Any]:
    initial_state = snapshot_adapter(model)
    run_dir = RUN_ROOT / run_label
    adapter_dir = ADAPTER_ROOT
    stage_results: list[dict[str, Any]] = []
    stage_scores: list[dict[str, Any]] = []
    adapter_state = initial_state

    for task_index, domain in enumerate(DOMAINS, start=1):
        attempt_dir = run_dir / f"task_{task_index:02d}_{domain}"
        attempt, adapter_state = train_domain(
            model, tokenizer, train_rows, domain, max_length, device,
            adapter_state, attempt_dir,
        )
        stage_results.append(attempt)
        target_adapter = adapter_dir / f"task_{task_index:02d}_{domain}"
        model.save_pretrained(target_adapter)
        model.config.use_cache = True
        score_path = run_dir / f"validation-after-task-{task_index}.jsonl"
        try:
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            eval_started = time.perf_counter()
            score = score_stage(
                model, tokenizer, validation_rows, task_index, device, max_length, score_path
            )
            if device == "cuda":
                torch.cuda.synchronize()
            score["validation_seconds"] = time.perf_counter() - eval_started
            score["peak_validation_vram_reserved_gib"] = (
                torch.cuda.max_memory_reserved() / 1024**3 if device == "cuda" else None
            )
        except BaseException as error:
            if not is_cuda_oom(error):
                raise
            raise PilotOOM(json.dumps({
                "domain": domain,
                "task_index": task_index,
                "phase": "validation",
                "device": device,
                "max_sequence_length": max_length,
                "peak_validation_vram_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                "peak_validation_vram_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
                "oom": True,
                "oom_message": str(error)[:1000],
            })) from error
        stage_scores.append(score)
        model.config.use_cache = False
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    return {
        "run_label": run_label,
        "device": device,
        "max_sequence_length": max_length,
        "model_id": MODEL_ID,
        "model_revision": revision,
        "task_order": list(DOMAINS),
        "training": stage_results,
        "validation": stage_scores,
    }


def main() -> None:
    if not TRAIN_FILE.exists() or not VALIDATION_FILE.exists():
        raise FileNotFoundError("Run prepare_clinc150.py before the pilot")
    if not torch.cuda.is_available():
        raise RuntimeError("The GTX 1650 is not visible to PyTorch; no CPU fallback was requested yet")
    free_gib = shutil.disk_usage(ROOT).free / 1024**3
    if free_gib < 5:
        raise RuntimeError(f"Only {free_gib:.1f} GiB free; stop before downloading model weights")

    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    ADAPTER_ROOT.mkdir(parents=True, exist_ok=True)
    capture_nvidia_smi(RUN_ROOT / "nvidia-smi-before-pilot.txt")
    set_seed(SEED)
    config = AutoConfig.from_pretrained(MODEL_ID)
    revision = getattr(config, "_commit_hash", None) or "main"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = create_lora_model(revision, torch.float16, "cuda")
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    unexpected_trainables: list[str] = []

    train_rows = read_jsonl(TRAIN_FILE)
    validation_rows = read_jsonl(VALIDATION_FILE)
    print(json.dumps({
        "model": MODEL_ID,
        "revision": revision,
        "gpu": torch.cuda.get_device_name(0),
        "vram_gib": torch.cuda.get_device_properties(0).total_memory / 1024**3,
        "disk_free_gib_before_model_download": free_gib,
        "trainable_lora_parameters": trainable,
        "trainable_non_lora_parameters": len(unexpected_trainables),
        "max_sequence_length": BASE_MAX_LENGTH,
        "domains": list(DOMAINS),
    }, indent=2), flush=True)

    attempts: list[dict[str, Any]] = []
    pilot_result: dict[str, Any] | None = None
    try:
        pilot_result = run_two_domain_pilot(
            model, tokenizer, train_rows, validation_rows,
            revision, "cuda", BASE_MAX_LENGTH, "gpu-seq128",
        )
        attempts.append({"run_label": "gpu-seq128", "completed": True})
    except PilotOOM as error:
        attempts.append({"run_label": "gpu-seq128", "completed": False,
                         "oom_details": str(error)})
        gc.collect()
        torch.cuda.empty_cache()
        # Rerun both domains at one shared length, with the original seeded adapter.
        del model
        gc.collect()
        torch.cuda.empty_cache()
        set_seed(SEED)
        model = create_lora_model(revision, torch.float16, "cuda")
        try:
            pilot_result = run_two_domain_pilot(
                model, tokenizer, train_rows, validation_rows,
                revision, "cuda", OOM_RETRY_LENGTH, "gpu-seq96",
            )
            attempts.append({"run_label": "gpu-seq96", "completed": True})
        except PilotOOM as retry_error:
            attempts.append({"run_label": "gpu-seq96", "completed": False,
                             "oom_details": str(retry_error)})
            print("GPU pilot OOM persisted at 96 tokens; restarting the whole pilot on CPU.", flush=True)
            del model
            gc.collect()
            torch.cuda.empty_cache()
            set_seed(SEED)
            model = create_lora_model(revision, torch.float32, "cpu")
            pilot_result = run_two_domain_pilot(
                model, tokenizer, train_rows, validation_rows,
                revision, "cpu", OOM_RETRY_LENGTH, "cpu-seq96",
            )
            attempts.append({"run_label": "cpu-seq96", "completed": True})

    pilot_result["attempts"] = attempts
    pilot_result["status"] = "completed"
    pilot_result["pilot_passed"] = (
        pilot_result["device"] == "cuda"
        and len(pilot_result["training"]) == 2
        and all(not item["oom"] for item in pilot_result["training"])
        and len(pilot_result["validation"]) == 2
        and all(item["validation_examples"] > 0 for item in pilot_result["validation"])
    )
    pilot_result["pass_criteria"] = (
        "Both CUDA domain trainings completed without OOM and both stage validation sets were scored."
    )
    pilot_result["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    pilot_result["settings"] = {
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "target_modules": TARGET_MODULES,
        "trainable_base_weights": "none",
        "precision": "fp16" if pilot_result["device"] == "cuda" else "fp32_cpu_fallback",
        "per_device_batch_size": PER_DEVICE_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "epochs_per_domain": EPOCHS_PER_DOMAIN,
        "optimizer": "AdamW (torch)",
        "learning_rate": LEARNING_RATE,
        "lr_scheduler": "constant; zero warmup",
        "weight_decay": 0.0,
        "gradient_clipping": MAX_GRAD_NORM,
        "gradient_checkpointing": True,
        "seed": SEED,
        "max_new_tokens": MAX_NEW_TOKENS,
        "decoding": "greedy (do_sample=False, num_beams=1)",
        "chat_template": "Qwen tokenizer template around the exact Step 3 user prompt body",
        "prompt_loss": "masked; loss is on assistant label and end-of-turn tokens",
        "trainable_adapter_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "trainable_non_adapter_parameters": 0,
        "padding": "dynamic within batch",
        "overlength_policy": "raise an error; do not truncate prompt or target",
    }
    metrics_path = RUN_ROOT / "pilot_metrics.json"
    metrics_path.write_text(json.dumps(pilot_result, indent=2) + "\n", encoding="utf-8")
    capture_nvidia_smi(RUN_ROOT / "nvidia-smi-after-pilot.txt")
    print(json.dumps({
        "status": pilot_result["status"],
        "device": pilot_result["device"],
        "max_sequence_length": pilot_result["max_sequence_length"],
        "training": pilot_result["training"],
        "validation": pilot_result["validation"],
        "metrics_file": str(metrics_path),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
