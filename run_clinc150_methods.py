"""Run sequential LoRA, reservoir replay LoRA, and O-LoRA on CLINC150."""
from __future__ import annotations

import csv
import argparse
import gc
import json
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from torch.utils.data import RandomSampler, SequentialSampler
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

from experiment_io import TASK_ORDER, accepted_prediction, available_intents, format_prompt, format_target, is_correct_prediction, normalize_generated_output
from pilot_clinc150 import (ROOT, MODEL_ID, SEED, BASE_MAX_LENGTH, LORA_RANK, LORA_ALPHA,
    LORA_DROPOUT, TARGET_MODULES, PER_DEVICE_BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS,
    EPOCHS_PER_DOMAIN, LEARNING_RATE, MAX_GRAD_NORM, MAX_NEW_TOKENS,
    PromptTargetDataset, DynamicPadCollator, read_jsonl, capture_nvidia_smi)

OUT = ROOT / "run-logs" / "full-sequence"
ADAPTERS = ROOT / "artifacts" / "full-sequence"
REPLAY_CAPACITY = 300
REPLAY_CURRENT = 6
REPLAY_OLD = 2
O_LORA_LAMBDA = 0.5


class OrderedTrainer(Trainer):
    """Keep the prebuilt replay proportions aligned to accumulation groups."""
    method = ""
    adapter_name = ""

    def _get_train_sampler(self, train_dataset=None):
        active_dataset = train_dataset or self.train_dataset
        if self.method == "replay" and len(active_dataset) == 2000:
            return SequentialSampler(active_dataset)
        return RandomSampler(active_dataset)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs.loss
        if self.method == "olora":
            params = dict(model.named_parameters())
            current = [p for n, p in params.items() if f"lora_A.{self.adapter_name}.weight" in n]
            penalties = []
            for name, new_a in ((n, p) for n, p in params.items() if f"lora_A.{self.adapter_name}.weight" in n):
                prefix = name.split("lora_A.", 1)[0]
                for old_name, old_a in params.items():
                    if (old_name.startswith(prefix + "lora_A.") and old_name.endswith(".weight")
                        and old_name != name and ".lora_A.default." not in old_name):
                        gram = old_a.detach().float().T @ new_a.float()
                        penalties.append(gram.square().mean())
            if penalties:
                loss = loss + O_LORA_LAMBDA * torch.stack(penalties).sum()
        return (loss, outputs) if return_outputs else loss


def tokenizer_dataset(rows, tokenizer):
    return PromptTargetDataset(rows, tokenizer, BASE_MAX_LENGTH)


def new_model(revision, tokenizer):
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=revision,
        torch_dtype=torch.float16, low_cpu_mem_usage=True)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=LORA_RANK, lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT, target_modules=TARGET_MODULES, bias="none", task_type="CAUSAL_LM"))
    model.to("cuda")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model


def add_adapter(model, name):
    model.add_adapter(name, LoraConfig(r=LORA_RANK, lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT, target_modules=TARGET_MODULES, bias="none", task_type="CAUSAL_LM"))
    # PeftModel.add_adapter does not apply the FP32 adapter upcast performed by
    # get_peft_model. Keep trainable LoRA weights in FP32 for FP16 GradScaler.
    for param_name, param in model.named_parameters():
        if (f"lora_A.{name}.weight" in param_name or f"lora_B.{name}.weight" in param_name) \
            and param.dtype in (torch.float16, torch.bfloat16):
            param.data = param.data.float()


def train_one(model, tokenizer, rows, method, domain, task_index, replay_rows=None, adapter_name="default"):
    if method == "replay" and replay_rows:
        # 250 effective batches: exactly 1,500 current + 500 uniformly sampled replay rows.
        rng = random.Random(SEED + task_index)
        current_order = rows.copy()
        rng.shuffle(current_order)
        stream = []
        for offset in range(0, len(current_order), REPLAY_CURRENT):
            group = current_order[offset:offset + REPLAY_CURRENT]
            group += [rng.choice(replay_rows) for _ in range(REPLAY_OLD)]
            rng.shuffle(group)
            stream.extend(group)
        dataset = tokenizer_dataset(stream, tokenizer)
        current_count, replay_count = 1500, len(stream) - 1500
    else:
        dataset = tokenizer_dataset(rows, tokenizer)
        current_count, replay_count = len(rows), 0

    if method == "olora":
        active = [f"task_{i:02d}" for i in range(1, task_index + 1)]
        model.base_model.set_adapter(active)
        for name, param in model.named_parameters():
            param.requires_grad = (
                f"lora_A.{adapter_name}." in name or f"lora_B.{adapter_name}." in name
            )
    else:
        model.set_adapter("default")
        for name, param in model.named_parameters():
            param.requires_grad = "lora_" in name

    args = TrainingArguments(output_dir=str(OUT / "work" / method / f"task_{task_index:02d}"),
        num_train_epochs=EPOCHS_PER_DOMAIN, per_device_train_batch_size=1,
        gradient_accumulation_steps=8, learning_rate=LEARNING_RATE,
        lr_scheduler_type="constant", warmup_steps=0, weight_decay=0.0,
        optim="adamw_torch", max_grad_norm=MAX_GRAD_NORM, fp16=True,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_strategy="steps", logging_steps=25, save_strategy="no", eval_strategy="no",
        report_to=[], seed=SEED, data_seed=SEED, dataloader_num_workers=0,
        remove_unused_columns=True, disable_tqdm=False)
    trainer = OrderedTrainer(model=model, args=args, train_dataset=dataset,
        data_collator=DynamicPadCollator(tokenizer.pad_token_id))
    trainer.method, trainer.adapter_name = method, adapter_name
    stage = OUT / method / f"task_{task_index:02d}_{domain}"
    stage.mkdir(parents=True, exist_ok=True)
    capture_nvidia_smi(stage / "nvidia-smi-before.txt")
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    result = trainer.train()
    torch.cuda.synchronize()
    duration = time.perf_counter() - start
    record = {"method": method, "task_index": task_index, "domain": domain,
        "current_examples": current_count, "replay_examples": replay_count,
        "optimizer_updates": trainer.state.global_step, "training_loss": result.training_loss,
        "training_seconds": duration, "peak_vram_allocated_gib": torch.cuda.max_memory_allocated()/1024**3,
        "peak_vram_reserved_gib": torch.cuda.max_memory_reserved()/1024**3}
    capture_nvidia_smi(stage / "nvidia-smi-after.txt")
    del trainer, dataset; gc.collect(); torch.cuda.empty_cache()
    return record


def generate_score(model, tokenizer, rows, task_index, method, split):
    seen = set(TASK_ORDER[:task_index])
    eval_rows = [r for r in rows if r["domain"] in seen]
    out = OUT / method / f"{split}-after-task-{task_index:02d}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    stats = {d: {"correct": 0, "total": 0, "invalid": 0} for d in TASK_ORDER[:task_index]}
    invalid = correct = 0
    completed = []
    if out.exists():
        # Keep only a verified prefix so an interrupted score can resume without losing work.
        with out.open("r", encoding="utf-8") as existing_file:
            for index, line in enumerate(existing_file):
                if index >= len(eval_rows):
                    break
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    break
                row = eval_rows[index]
                expected = (method, split, task_index, row["domain"], row["text"], row["intent"])
                actual = (item.get("method"), item.get("split"), item.get("task_index"),
                    item.get("domain"), item.get("text"), item.get("target"))
                if actual != expected or "correct" not in item or "accepted_prediction" not in item:
                    break
                completed.append(item)
        for item in completed:
            item_stats = stats[item["domain"]]
            item_stats["total"] += 1
            item_stats["correct"] += int(item["correct"])
            item_stats["invalid"] += int(item["accepted_prediction"] is None)
            correct += int(item["correct"])
            invalid += int(item["accepted_prediction"] is None)
    model.eval()
    with out.open("w", encoding="utf-8", newline="\n") as f:
        for item in completed:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
        f.flush()
        for row in eval_rows[len(completed):]:
            ids = tokenizer.apply_chat_template([{"role":"user", "content":format_prompt(row["text"])}],
                tokenize=True, add_generation_prompt=True)
            if len(ids) > BASE_MAX_LENGTH:
                raise ValueError(f"Evaluation prompt length {len(ids)} exceeds {BASE_MAX_LENGTH}")
            input_ids = torch.tensor([ids], dtype=torch.long, device="cuda")
            try:
                with torch.inference_mode():
                    generated = model.generate(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                        max_new_tokens=MAX_NEW_TOKENS, do_sample=False, num_beams=1,
                        temperature=1.0, top_p=1.0, top_k=50,
                        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
            except RuntimeError as exc:
                message = str(exc).lower()
                if "out of memory" not in message and "cublas_status_execution_failed" not in message:
                    raise
                del input_ids
                gc.collect(); torch.cuda.empty_cache()
                input_ids = torch.tensor([ids], dtype=torch.long, device="cuda")
                with torch.inference_mode():
                    generated = model.generate(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                        max_new_tokens=MAX_NEW_TOKENS, do_sample=False, num_beams=1,
                        temperature=1.0, top_p=1.0, top_k=50,
                        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
            raw = tokenizer.decode(generated[0, input_ids.shape[1]:], skip_special_tokens=True)
            pred = accepted_prediction(raw, task_index)
            ok = is_correct_prediction(raw, row["intent"], task_index)
            invalid += int(pred is None); correct += int(ok)
            item = stats[row["domain"]]; item["total"] += 1; item["correct"] += int(ok); item["invalid"] += int(pred is None)
            f.write(json.dumps({"method":method,"split":split,"task_index":task_index,
                "domain":row["domain"],"text":row["text"],"target":row["intent"],
                "raw_output":raw,"normalized_output":normalize_generated_output(raw),
                "accepted_prediction":pred,"correct":ok},ensure_ascii=False)+"\n")
            f.flush()
            del generated, input_ids
    model.train()
    for value in stats.values():
        value["accuracy"] = value["correct"] / value["total"]
    return {"method":method,"split":split,"task_index":task_index,"examples":len(eval_rows),
        "correct":correct,"accuracy":correct/len(eval_rows),"invalid_outputs":invalid,
        "per_domain":stats,"predictions":str(out.relative_to(ROOT))}


def reservoir_update(buffer, rows, seen_count, rng):
    for row in rows:
        seen_count += 1
        if len(buffer) < REPLAY_CAPACITY:
            buffer.append(row)
        else:
            j = rng.randrange(seen_count)
            if j < REPLAY_CAPACITY:
                buffer[j] = row
    return seen_count


def save_adapter(model, folder, name):
    folder.mkdir(parents=True, exist_ok=True)
    state = {k:v.detach().cpu().contiguous()
        for k,v in get_peft_model_state_dict(model, adapter_name=name).items()}
    torch.save(state, folder / "adapter_model.bin")
    (folder / "adapter_config.json").write_text(json.dumps({"base_model":MODEL_ID,"rank":4,"alpha":8,
        "dropout":0.05,"target_modules":TARGET_MODULES,"adapter_name":name},indent=2),encoding="utf-8")


def run_method(method, revision, tokenizer, train, valid, test):
    set_seed(SEED)
    model = new_model(revision, tokenizer)
    buffer, seen_count, train_records, val_records, test_records = [], 0, [], [], []
    for task_index, domain in enumerate(TASK_ORDER, 1):
        current = [r for r in train if r["domain"] == domain]
        if method == "olora":
            name=f"task_{task_index:02d}"; add_adapter(model,name)
            if task_index == 1:
                model.set_adapter(name)
                model.delete_adapter("default")
            # Active stack is all prior plus current; only current adapter receives gradients.
            record = train_one(model,tokenizer,current,method,domain,task_index,
                adapter_name=name)
            save_adapter(model, ADAPTERS / method / name, name)
            model.base_model.set_adapter([f"task_{i:02d}" for i in range(1,task_index+1)])
        elif method == "replay":
            record = train_one(model,tokenizer,current,method,domain,task_index,
                replay_rows=buffer if task_index > 1 else None)
            save_adapter(model, ADAPTERS / method / f"task_{task_index:02d}_{domain}", "default")
        else:
            record = train_one(model,tokenizer,current,method,domain,task_index)
            save_adapter(model, ADAPTERS / method / f"task_{task_index:02d}_{domain}", "default")
        train_records.append(record)
        write_results(method,train_records,val_records,test_records)
        val_records.append(generate_score(model,tokenizer,valid,task_index,method,"validation"))
        write_results(method,train_records,val_records,test_records)
        test_records.append(generate_score(model,tokenizer,test,task_index,method,"test"))
        # Reservoir receives only the just-completed domain's training rows.
        if method == "replay":
            seen_count = reservoir_update(buffer,current,seen_count,random.Random(SEED + task_index))
            # Persist exact buffer and its provenance for reproducibility.
            (OUT / method).mkdir(parents=True,exist_ok=True)
            with (OUT / method / f"buffer-after-task-{task_index:02d}.jsonl").open("w",encoding="utf-8") as f:
                for r in buffer: f.write(json.dumps(r,ensure_ascii=False)+"\n")
        # Stage metrics are durable after every boundary.
        write_results(method,train_records,val_records,test_records)
    if method == "olora":
        size = sum(p.stat().st_size for p in (ADAPTERS / method).rglob("*" ) if p.is_file())
        (OUT / method / "adapter_storage_bytes.json").write_text(json.dumps({"combined_adapter_bytes":size},indent=2),encoding="utf-8")
    del model; gc.collect(); torch.cuda.empty_cache()
    return train_records,val_records,test_records


def run_sequential_resume(start_task, revision, tokenizer, train, valid, test):
    """Resume a sequential run from its saved adapter at a completed task boundary."""
    if not 1 <= start_task <= len(TASK_ORDER):
        raise ValueError("Resume task must be between 1 and 10")
    folder = OUT / "sequential"
    checkpoint = ADAPTERS / "sequential" / f"task_{start_task:02d}_{TASK_ORDER[start_task-1]}" / "adapter_model.bin"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Sequential checkpoint not found: {checkpoint}")
    model = new_model(revision, tokenizer)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    set_peft_model_state_dict(model, state, adapter_name="default")
    train_records = json.loads((folder / "training_metrics.json").read_text(encoding="utf-8"))
    val_records = json.loads((folder / "validation_metrics.json").read_text(encoding="utf-8"))
    test_records = json.loads((folder / "test_metrics.json").read_text(encoding="utf-8"))
    if len(train_records) < start_task or len(val_records) < start_task:
        raise ValueError("Saved training and validation metrics do not reach the requested resume task")
    if len(test_records) >= start_task:
        raise ValueError("The requested resume task already has a complete test metric")

    # Re-score the interrupted test stage from the beginning; generate_score replaces its partial JSONL.
    test_records.append(generate_score(model, tokenizer, test, start_task, "sequential", "test"))
    write_results("sequential", train_records, val_records, test_records)
    for task_index in range(start_task + 1, len(TASK_ORDER) + 1):
        domain = TASK_ORDER[task_index - 1]
        current = [r for r in train if r["domain"] == domain]
        record = train_one(model, tokenizer, current, "sequential", domain, task_index)
        train_records.append(record)
        save_adapter(model, ADAPTERS / "sequential" / f"task_{task_index:02d}_{domain}", "default")
        write_results("sequential", train_records, val_records, test_records)
        val_records.append(generate_score(model, tokenizer, valid, task_index, "sequential", "validation"))
        write_results("sequential", train_records, val_records, test_records)
        test_records.append(generate_score(model, tokenizer, test, task_index, "sequential", "test"))
        write_results("sequential", train_records, val_records, test_records)
    del model
    gc.collect(); torch.cuda.empty_cache()


def run_replay_resume(start_task, revision, tokenizer, train, valid, test):
    """Resume replay from a completed task boundary, including its exact reservoir."""
    if not 1 <= start_task < len(TASK_ORDER):
        raise ValueError("Replay resume task must be between 1 and 9")
    folder = OUT / "replay"
    domain = TASK_ORDER[start_task - 1]
    checkpoint = ADAPTERS / "replay" / f"task_{start_task:02d}_{domain}" / "adapter_model.bin"
    buffer_path = folder / f"buffer-after-task-{start_task:02d}.jsonl"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Replay checkpoint not found: {checkpoint}")
    if not buffer_path.is_file():
        raise FileNotFoundError(f"Replay reservoir snapshot not found: {buffer_path}")

    train_records = json.loads((folder / "training_metrics.json").read_text(encoding="utf-8"))
    val_records = json.loads((folder / "validation_metrics.json").read_text(encoding="utf-8"))
    test_records = json.loads((folder / "test_metrics.json").read_text(encoding="utf-8"))
    if any(len(records) < start_task for records in (train_records, val_records, test_records)):
        raise ValueError("Replay metrics do not contain complete train, validation, and test results through the resume task")

    buffer = read_jsonl(buffer_path)
    if len(buffer) != REPLAY_CAPACITY:
        raise ValueError(f"Expected {REPLAY_CAPACITY} replay examples in {buffer_path}, found {len(buffer)}")
    # Each prior domain has 1,500 eligible rows; task 1 has no prior reservoir.
    seen_count = start_task * 1500
    model = new_model(revision, tokenizer)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    set_peft_model_state_dict(model, state, adapter_name="default")

    for task_index in range(start_task + 1, len(TASK_ORDER) + 1):
        domain = TASK_ORDER[task_index - 1]
        current = [r for r in train if r["domain"] == domain]
        record = train_one(model, tokenizer, current, "replay", domain, task_index,
            replay_rows=buffer)
        save_adapter(model, ADAPTERS / "replay" / f"task_{task_index:02d}_{domain}", "default")
        train_records.append(record)
        write_results("replay", train_records, val_records, test_records)
        val_records.append(generate_score(model, tokenizer, valid, task_index, "replay", "validation"))
        write_results("replay", train_records, val_records, test_records)
        test_records.append(generate_score(model, tokenizer, test, task_index, "replay", "test"))
        seen_count = reservoir_update(buffer, current, seen_count, random.Random(SEED + task_index))
        with (folder / f"buffer-after-task-{task_index:02d}.jsonl").open("w", encoding="utf-8") as stream:
            for row in buffer:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        write_results("replay", train_records, val_records, test_records)
    del model
    gc.collect(); torch.cuda.empty_cache()


def run_zero_shot(revision, tokenizer, valid, test):
    """Score the untouched base model once, with all 150 labels available."""
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=revision,
        torch_dtype=torch.float16, low_cpu_mem_usage=True).to("cuda")
    model.config.use_cache = True
    results = {}
    for split, rows in (("validation", valid), ("test", test)):
        results[split] = generate_score(model, tokenizer, rows, len(TASK_ORDER), "zero_shot", split)
    folder = OUT / "zero_shot"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    with (folder / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["split", "examples", "correct", "accuracy", "invalid_outputs"])
        writer.writeheader()
        for split, result in results.items():
            writer.writerow({"split": split, **{key: result[key] for key in ("examples", "correct", "accuracy", "invalid_outputs")}})
    del model; gc.collect(); torch.cuda.empty_cache()


def write_results(method, trains, vals, tests):
    folder=OUT/method; folder.mkdir(parents=True,exist_ok=True)
    for name, records in (("training",trains),("validation",vals),("test",tests)):
        with (folder/f"{name}_metrics.json").open("w",encoding="utf-8") as f:
            json.dump(records,f,indent=2)
        with (folder/f"{name}_metrics.csv").open("w",encoding="utf-8",newline="") as f:
            fields=[k for k in records[0] if k!="per_domain"] if records else []
            writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader()
            for r in records: writer.writerow({k:r[k] for k in fields})
    with (folder/"test_accuracy_matrix.csv").open("w",encoding="utf-8",newline="") as f:
        cols=["task_index"]+list(TASK_ORDER); writer=csv.DictWriter(f,fieldnames=cols); writer.writeheader()
        for record in tests:
            row={"task_index":record["task_index"]}
            row.update({domain:record["per_domain"].get(domain,{}).get("accuracy","") for domain in TASK_ORDER})
            writer.writerow(row)
    if len(tests)==10:
        domain_acc=[tests[i]["per_domain"][TASK_ORDER[i]]["accuracy"] for i in range(10)]
        final=tests[-1]; forgetting=[]; bwt=[]
        for j,d in enumerate(TASK_ORDER):
            best=max(t["per_domain"][d]["accuracy"] for t in tests[j:])
            end=final["per_domain"][d]["accuracy"]
            immediate=tests[j]["per_domain"][d]["accuracy"]
            forgetting.append(best-end); bwt.append(end-immediate)
        summary={"final_accuracy":final["accuracy"],"final_correct":final["correct"],
            "final_examples":final["examples"],"final_invalid_outputs":final["invalid_outputs"],
            "forgetting_by_domain":dict(zip(TASK_ORDER,forgetting)),"mean_forgetting":sum(forgetting)/10,
            "backward_transfer_by_domain":dict(zip(TASK_ORDER,bwt)),"backward_transfer":sum(bwt)/10,
            "peak_vram_gib_by_task":{r["domain"]:{"allocated":r["peak_vram_allocated_gib"],"reserved":r["peak_vram_reserved_gib"]} for r in trains},
            "training_seconds_by_task":{r["domain"]:r["training_seconds"] for r in trains}}
        (folder/"final_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", choices=("sequential", "replay", "olora"),
        default=("sequential", "replay", "olora"))
    parser.add_argument("--zero-shot-only", action="store_true")
    parser.add_argument("--resume-sequential-from", type=int, default=0,
        help="Resume sequential training and evaluation from an existing task checkpoint (1-10)")
    parser.add_argument("--resume-replay-from", type=int, default=0,
        help="Resume replay training and evaluation from an existing task checkpoint (1-9)")
    args = parser.parse_args()
    selected_methods = args.methods
    if not torch.cuda.is_available(): raise RuntimeError("CUDA GPU required; no automatic method change")
    OUT.mkdir(parents=True,exist_ok=True); ADAPTERS.mkdir(parents=True,exist_ok=True)
    capture_nvidia_smi(OUT/"nvidia-smi-before.txt")
    config=AutoConfig.from_pretrained(MODEL_ID); revision=getattr(config,"_commit_hash",None) or "main"
    tokenizer=AutoTokenizer.from_pretrained(MODEL_ID,revision=revision)
    train=read_jsonl(ROOT/"data"/"prepared"/"train.jsonl")
    valid=read_jsonl(ROOT/"data"/"prepared"/"validation.jsonl")
    test=read_jsonl(ROOT/"data"/"prepared"/"test.jsonl")
    meta={"model_id":MODEL_ID,"revision":revision,"task_order":list(TASK_ORDER),"seed":SEED,
        "max_sequence_length":BASE_MAX_LENGTH,"replay_capacity":REPLAY_CAPACITY,
        "replay_microbatch_ratio":"6 current : 2 replay; effective batch via 8 accumulation steps",
        "olora_lambda":O_LORA_LAMBDA,"created_at_utc":datetime.now(timezone.utc).isoformat()}
    (OUT/"run_config.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    if args.zero_shot_only:
        run_zero_shot(revision, tokenizer, valid, test)
        capture_nvidia_smi(OUT/"nvidia-smi-after-zero-shot.txt")
        return
    if args.resume_sequential_from:
        if tuple(selected_methods) != ("sequential",):
            raise ValueError("--resume-sequential-from requires --methods sequential")
        run_sequential_resume(args.resume_sequential_from, revision, tokenizer, train, valid, test)
        capture_nvidia_smi(OUT/"nvidia-smi-after.txt")
        return
    if args.resume_replay_from:
        if "replay" not in selected_methods:
            raise ValueError("--resume-replay-from requires replay in --methods")
        run_replay_resume(args.resume_replay_from, revision, tokenizer, train, valid, test)
        for method in selected_methods:
            if method == "replay":
                continue
            print(f"Starting full method: {method}", flush=True)
            run_method(method, revision, tokenizer, train, valid, test)
        capture_nvidia_smi(OUT/"nvidia-smi-after.txt")
        return
    for method in selected_methods:
        print(f"Starting full method: {method}",flush=True)
        run_method(method,revision,tokenizer,train,valid,test)
    capture_nvidia_smi(OUT/"nvidia-smi-after.txt")

if __name__=="__main__": main()
