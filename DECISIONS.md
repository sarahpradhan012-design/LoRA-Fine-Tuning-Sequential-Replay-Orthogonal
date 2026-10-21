# CLINC150 continual-learning setup

## Fixed experiment choices

- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Dataset: CLINC150 full version, excluding OOS examples from the main experiment
- Task order: `auto_and_commute, banking, credit_cards, home, kitchen_and_dining, meta, small_talk, travel, utility, work`
- Methods: frozen-base sequential LoRA, replay LoRA, O-LoRA, plus zero-shot reference
- Hardware interpretation: results describe this single GTX 1650 4 GB setup only

## Step 1 status

This is a Windows setup. Use ordinary FP16 LoRA first; QLoRA and full fine-tuning are out of scope. The environment targets Python 3.11 and the PyTorch 2.5.1 CUDA 12.1 wheel. The installed NVIDIA driver reported by `nvidia-smi` is 529.04; its `CUDA Version: 12.0` field is not the GPU compute capability. NVIDIA lists driver 528.33 or newer for CUDA 12.x minor-version compatibility on Windows, so this driver is new enough for the PyTorch cu121 wheel. The definitive check is `torch.cuda.is_available()` and successful GPU allocation below.

C: had 144.3 GB free before environment installation and 126.2 GB afterward. The UCI dataset and Qwen model snapshot are now downloaded. No API key is needed for local use of these public resources.

## Environment creation

From PowerShell in this folder:

```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe" -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python .\verify_setup.py
```

If PowerShell blocks activation, run the environment's Python directly, for example `.\.venv\Scripts\python.exe -m pip ...`.

## Run records

Before each experiment run, capture the GPU and package details:

```powershell
New-Item -ItemType Directory -Force .\run-logs | Out-Null
nvidia-smi | Out-File -Encoding utf8 .\run-logs\nvidia-smi-$(Get-Date -Format yyyyMMdd-HHmmss).txt
.\.venv\Scripts\python.exe -m pip freeze | Out-File -Encoding utf8 .\run-logs\pip-freeze-$(Get-Date -Format yyyyMMdd-HHmmss).txt
```

## Step 2: CLINC150 data

The UCI full release is saved at `data/raw/clinc150-uci.zip`. The prepared files preserve the official train, validation, and test partitions while excluding OOS rows:

- `data/prepared/train.jsonl` — 15,000 examples
- `data/prepared/validation.jsonl` — 3,000 examples
- `data/prepared/test.jsonl` — 4,500 examples
- `data/manifest.json` — source details, stable intent/domain/label maps, split counts, seed, and overlap report

Each record uses the intent name in the `intent` field as its target. `label_id` is a stable auxiliary index. The fixed seed is 42; no sampling or shuffling is performed during preparation. Rebuild and recheck with:

```powershell
.\.venv\Scripts\python.exe .\prepare_clinc150.py
```

**Source split overlap:** the official train and validation portions contain the identical `(text, intent)` row `("hey what's up", "greeting")`. There are also identical utterance texts across splits with different intent labels. The official partitions and requested counts are preserved, so the manifest records this as a failed leakage check. No replay buffer has been created. Do not treat the split-leakage requirement as satisfied unless this source issue is resolved or explicitly accepted.

The Step 1 CUDA check passed before the UCI data download. Verify `verify_setup.py` again before starting model download if the driver or environment has changed.

## Step 3: Model input and output

Implementation: `experiment_io.py`. Step 4 passes the returned prompt body through Qwen's model chat template as described below.

Every example uses this exact prompt, with the source utterance inserted verbatim:

```text
Classify this user request. Reply with one intent label only.
Request: {utterance}
```

The training target is the exact canonical intent name from the data record's `intent` field. The manifest's stable numeric `label_id` is only an auxiliary index and is not the generated target. `format_prompt`, `format_model_prompt`, `format_target`, `available_intents`, `normalize_generated_output`, `accepted_prediction`, and `is_correct_prediction` centralize the contract. `task_index` is 1-based and inclusive; stage `t` accepts only the labels in the first `t` domains of the fixed order. Invalid predictions return `None` from `accepted_prediction` and count as incorrect.

At evaluation, output normalization trims outer whitespace and trailing Unicode punctuation, while preserving `_` because underscores are part of canonical labels. Matching is case-sensitive and exact. Empty, malformed, unknown, and future-task labels are invalid; they receive no manual correction and count as incorrect. The prompt contains no domain, task number, or label inventory.

## Step 4: Two-domain memory pilot

`pilot_clinc150.py` runs sequential LoRA on `auto_and_commute` then `banking`, carrying the same adapter forward between domains. It uses the supplied rank, alpha, dropout, target modules, FP16, batch size, gradient accumulation, sequence length, epochs, AdamW, learning rate, clipping, and gradient checkpointing. It checks that only LoRA parameters are trainable. The nominal update estimate is `ceil(1,500 / 8) = 188` per domain; the observed Trainer count is recorded below.

The Qwen model card demonstrates its tokenizer chat template for instruction prompts ([model card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)). The user-visible prompt body stays exactly as specified in Step 3; the Qwen chat template supplies the model's role and assistant-generation markers around that body. Training loss is masked on the prompt tokens and applied to the canonical assistant label plus its end-of-turn token. This trains the requested answer while retaining the model's instruction format.

The max-length limit includes the Qwen chat wrapper, the exact prompt body, and (for training) the target and end-of-turn token. Batches use dynamic padding to their longest sequence to avoid needless padding memory. If a sequence exceeds the cap, preparation stops with an error instead of silently truncating the request or target; this preserves the exact input and canonical label. Validation runs after each domain: 300 `auto_and_commute` examples after task 1, then 600 examples across both seen domains after task 2. The script saves per-example raw and normalized outputs, accepted label, correctness, stage metrics, per-domain scores, model revision, adapter snapshots, per-domain training time, peak allocated/reserved VRAM, and `nvidia-smi` before/after snapshots under `run-logs/pilot/` and `artifacts/pilot/`.

If either GPU stage OOMs at length 128, the script restarts the **whole** two-domain pilot at length 96. This keeps the sequence length consistent between domains. If either stage OOMs again at 96, it restarts the whole pilot on CPU in FP32 at length 96; FP16 is not used on the CPU fallback. Attempts and the selected final settings are recorded in `pilot_metrics.json`. No model or experiment setting is silently changed.

The script pins the resolved Hugging Face model commit in the run metrics, uses deterministic greedy decoding with `max_new_tokens=16`, and seeds both model setup and training/shuffling with 42. These were left unspecified by the pilot table: the resolved commit makes the public model snapshot reproducible; greedy decoding avoids sampling variance; 16 generated tokens are ample for the short intent labels while limiting runaway responses; the seed follows Step 2's reproducibility rule. The script now passes neutral temperature/top-p/top-k values; they do not affect greedy decoding. The completed pilot preceded that code cleanup and emitted a warning that the model's sampling defaults were ignored because `do_sample=False`; its greedy predictions are unaffected. Each domain gets a fresh AdamW optimizer with zero weight decay and no warmup; the learning-rate scheduler is constant at `1e-4` so the supplied learning rate remains fixed for its single epoch. The LoRA adapter itself is carried between domains, but optimizer moments are reset at the task boundary so banking starts a fresh task-specific optimizer pass.

Run after CUDA and Step 2 checks:

```powershell
.\.venv\Scripts\python.exe .\pilot_clinc150.py
```

### Pilot result (2026-10-19)

The pilot **passed the stated memory/score criteria** at FP16 and max length 128. Both domains completed with no OOM, and both validation stages produced scored outputs. No 96-token retry or CPU fallback was used. The run pinned model revision `7ae557604adf67be50417f59c2c2f167def9a775`.

| Domain / stage | Training time | Updates | Training loss | Peak CUDA allocated / reserved | Validation result |
|---|---:|---:|---:|---:|---:|
| `auto_and_commute` | 17m 9.7s | 187 | 1.2056 | 1.080 / 1.713 GiB | 240/300 correct (80.00%); 48 invalid |
| `banking` | 14m 14.1s | 187 | 0.5594 | 1.070 / 1.465 GiB | 251/300 correct (83.67%) on banking |
| After task 2, retained `auto_and_commute` | included above | — | — | — | 114/300 correct (38.00%) |
| After task 2, both seen domains | — | — | — | — | 365/600 correct (60.83%); 210 invalid |

VRAM is PyTorch's peak allocated/reserved CUDA memory; before/after `nvidia-smi` snapshots are saved separately. Transformers completed 187 optimizer updates per 1,500-example epoch, one below the nominal `ceil(1,500 / 8) = 188` estimate; the logged `global_step` is the run's authoritative count. The large retention drop after banking is a pilot finding, not a change to the training settings. The Step 2 overlap examples are in later domains; there was no recorded duplicate text within the two pilot domains. The global Step 2 overlap issue remains unresolved for the full ten-domain experiment.

## Steps 5–7: Full continual-learning sequence

`run_clinc150_methods.py` runs the three methods from a separately loaded copy of the pinned base model, in the fixed domain order. Each domain uses the pilot's LoRA, FP16, length, optimizer, and one-epoch settings. Checkpoints are saved after each domain; validation and official test splits are evaluated after every task on all domains seen so far. Test results are recorded for reporting only and do not select settings or checkpoints. Per-example generations are retained as JSONL; per-domain stage metrics and the full test accuracy matrix are CSV/JSON under `run-logs/full-sequence/<method>/`. The untouched base model is also scored once as a zero-shot reference with all 150 in-scope labels available; its validation and test generations are stored separately under `run-logs/full-sequence/zero_shot/`.

Sequential LoRA carries one `default` adapter from task to task, creating a new Trainer and AdamW optimizer per domain as in the pilot; examples are shuffled with the fixed Trainer seed. Replay LoRA uses a reservoir buffer capped at 300 training records. After each domain, its 1,500 train rows are streamed through standard reservoir sampling; validation and test records are never eligible. For each later domain, the 1,500 current rows are shuffled once and each appears once, grouped six at a time; two buffer draws with replacement are added to each group. Thus replay tasks contain 2,000 microbatches and 250 optimizer updates (effective batch eight under accumulation), while task 1 has no prior buffer and follows the same seeded shuffled one-epoch path as sequential LoRA. This interprets the requested 6:2 ratio per accumulated batch while preserving one exposure to every current training record.

O-LoRA creates a separate rank-4 adapter per domain, freezes earlier adapters, and activates all learned adapters together for evaluation without a domain ID. PEFT needs an initial adapter to wrap the base model, so the unused bootstrap adapter is removed after the first task adapter is added. Training adds the paper's squared cross-product orthogonality penalty between each previous and current query/value `lora_A` subspace, with coefficient λ=0.5 (reported in the paper); task loss updates only the current adapter. Each rank-by-rank cross-product is averaged over its entries before summing across layers and previous adapters; this keeps the regularizer's scale less dependent on rank. This follows O-LoRA's use of the column subspace of A and fixed prior adapters ([paper](https://aclanthology.org/2023.findings-emnlp.715/)). Adapter files are stored individually and their combined on-disk byte size is recorded. This is a PEFT implementation following the paper's method, not the authors' original code.

After ten checkpoints, each method writes final test accuracy, invalid-output count, per-domain forgetting (`max accuracy from the domain's learning checkpoint through task 10, minus task-10 accuracy`), mean forgetting, and backward transfer (`task-10 accuracy minus accuracy immediately after learning that domain`, averaged over domains) to `final_summary.json`. Peak allocated/reserved CUDA memory and training time are recorded per domain. The accuracy matrix has one checkpoint row and one domain column; cells for domains not yet learned are blank.

Run with:

```powershell
.\.venv\Scripts\python.exe .\run_clinc150_methods.py
```

If running the reference separately, use `--zero-shot-only`; it evaluates the base model without adapters and does not change the continual-learning models.

### Full-run progress (updated 2026-10-21)

Sequential training and evaluation are complete for all ten domains. Tasks 1–10 cumulative test accuracies were 76.67%, 56.33%, 45.85%, 34.78%, 25.96%, 25.52%, 18.03%, 24.64%, 20.05%, and 25.76%. Task 10 (`work`) training took 901.6 s and validation scored 775/3,000 (25.83%, 1,697 invalid). Its test evaluation first stopped with CUDA out-of-memory after 1,965 of 4,500 predictions; resumable per-record scoring preserved completed outputs, and the test evaluation completed with 1,159/4,500 correct (25.76%, 2,451 invalid). The sequential `final_summary.json` reports mean forgetting 60.78 percentage points and backward transfer −60.78 percentage points. The task-10 checkpoint is saved at `artifacts/full-sequence/sequential/task_10_work/`.

Recovery support in `run_clinc150_methods.py` reloads a saved adapter and resumes at a completed task boundary. All runs are now complete. The zero-shot reference scored 0/3,000 validation and 0/4,500 test examples correct, with every output invalid under exact-label scoring. Final results are:

| Method | Final test accuracy | Invalid outputs | Mean forgetting | Backward transfer |
|---|---:|---:|---:|---:|
| Sequential LoRA | 25.76% (1,159/4,500) | 2,451 | 60.78 pp | −60.78 pp |
| Replay LoRA | 57.07% (2,568/4,500) | 1,360 | 29.44 pp | −29.13 pp |
| O-LoRA | 28.91% (1,301/4,500) | 2,679 | 55.07 pp | −55.07 pp |

#### Per-domain accuracy across stages

The full test accuracy matrices report each checkpoint row against each learned domain column; blank cells are future domains not yet learned. Accuracy is stage-aware: an intent from a future task is incorrect at that checkpoint. Read the complete stage-by-stage matrices for [Sequential LoRA](run-logs/full-sequence/sequential/test_accuracy_matrix.csv), [Replay LoRA](run-logs/full-sequence/replay/test_accuracy_matrix.csv), and [O-LoRA](run-logs/full-sequence/olora/test_accuracy_matrix.csv). Values below show each domain's test accuracy immediately after learning it → after task 10, followed by forgetting and backward transfer. All values are percentages or percentage points (pp). Forgetting uses the best accuracy from the learning checkpoint through task 10; backward transfer is final accuracy minus the immediate post-learning accuracy.

| Domain | Sequential learned → final | Forgetting | BWT | Replay learned → final | Forgetting | BWT | O-LoRA learned → final | Forgetting | BWT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| auto_and_commute | 76.67 → 14.44 | 62.22 | −62.22 | 76.67 → 41.78 | 38.00 | −34.89 | 76.00 → 30.89 | 45.11 | −45.11 |
| banking | 80.22 → 13.33 | 66.89 | −66.89 | 80.67 → 66.00 | 14.67 | −14.67 | 76.89 → 24.67 | 52.22 | −52.22 |
| credit_cards | 87.11 → 14.22 | 72.89 | −72.89 | 87.78 → 55.11 | 32.67 | −32.67 | 82.67 → 24.22 | 58.44 | −58.44 |
| home | 85.78 → 18.89 | 66.89 | −66.89 | 82.67 → 55.11 | 27.56 | −27.56 | 84.22 → 22.67 | 61.56 | −61.56 |
| kitchen_and_dining | 84.67 → 10.67 | 74.00 | −74.00 | 88.22 → 48.00 | 40.22 | −40.22 | 79.56 → 14.00 | 65.56 | −65.56 |
| meta | 83.33 → 2.44 | 80.89 | −80.89 | 86.00 → 50.00 | 36.00 | −36.00 | 81.78 → 5.33 | 76.44 | −76.44 |
| small_talk | 86.89 → 1.11 | 85.78 | −85.78 | 81.56 → 34.00 | 47.56 | −47.56 | 83.56 → 4.00 | 79.56 | −79.56 |
| travel | 94.44 → 37.78 | 56.67 | −56.67 | 92.22 → 52.67 | 39.56 | −39.56 | 91.78 → 31.11 | 60.67 | −60.67 |
| utility | 96.89 → 55.33 | 41.56 | −41.56 | 96.22 → 78.00 | 18.22 | −18.22 | 96.89 → 45.78 | 51.11 | −51.11 |
| work | 89.33 → 89.33 | 0.00 | 0.00 | 90.00 → 90.00 | 0.00 | 0.00 | 86.44 → 86.44 | 0.00 | 0.00 |

Replay resumed from the task-8 adapter and 300-example reservoir snapshot. Task 9 (`utility`) took 1,127.2 s, scored 60.37% validation and 59.09% test. Task 10 (`work`) took 1,174.5 s, scored 58.30% validation and 57.07% test. O-LoRA initially stopped on its first optimizer update because dynamically added FP16 adapter weights were incompatible with GradScaler's gradient unscale. The `add_adapter` helper now upcasts only the new trainable LoRA A/B weights to FP32, matching PEFT's initial adapter behavior; after that fix, all ten O-LoRA domains trained and were evaluated. O-LoRA final test accuracy was 28.91%, mean forgetting 55.07 percentage points, and backward transfer −55.07 percentage points. The combined adapter storage is 11,148,250 bytes (~10.63 MiB). The complete test accuracy matrices, per-example outputs, training measurements, and summaries are under `run-logs/full-sequence/{sequential,replay,olora}/`.

## Decision log

Record each material implementation choice here with its reason. Fixed choices supplied by the experiment owner are marked as such; implementation choices record the selected behavior and its rationale.

| Date | Decision | Reason and effect |
|---|---|---|
| 2026-10-17 | Use Python 3.11.9 with PyTorch 2.5.1+cu121 and pinned Transformers/PEFT/Datasets/Accelerate/Evaluate versions. | This pinned Windows stack installed and passed a real CUDA allocation/matmul check on the driver available at setup. It gives a reproducible baseline. Python 3.11.9 was selected because it was the last 3.11 release with a full Windows installer and has wheels for the pinned stack; later 3.11 security-only releases do not ship Windows installers. |
| 2026-10-17 | Start with ordinary FP16 LoRA; exclude QLoRA and full fine-tuning. *(Fixed choice.)* | The project specification prioritizes a simple Windows setup and identifies 4 GB VRAM as unsuitable for full fine-tuning. Do not add quantization complexity unless FP16 LoRA proves infeasible and the experiment owner revises the plan. |
| 2026-10-19 | Use only UCI CLINC150 `data_full.json`; exclude OOS rows and keep official train/val/test partitions. *(Fixed choice.)* | This matches the defined 150-intent experiment and keeps the separate rejection-detection task out of scope. |
| 2026-10-19 | Use seed 42, sorted intent names for stable auxiliary label IDs, and do not shuffle or sample during preparation. | A fixed seed makes any later sampling reproducible; no randomness is needed for this full-data preparation. Alphabetical IDs are stable, while the model's target remains the intent string. |
| 2026-10-19 | Keep official split files unchanged and record duplicate-text collisions instead of silently deleting or moving rows. | A train/validation row is identical in the source. Removing it would break the required per-intent count; moving it would alter the official partition. The manifest reports that the leakage condition therefore fails pending an explicit resolution. |
| 2026-10-19 | Use the shared two-line prompt for every task and emit the canonical intent name. *(Fixed choice.)* | Omitting domain/task cues and future label inventories prevents task metadata from entering the model input and avoids a long label list on every example. The intent name is the unambiguous target already present in CLINC150. |
| 2026-10-19 | Gate predictions by the cumulative seen-intent set at each stage; compare exact, case-sensitive strings. Use a 1-based inclusive task index. | A future-task label must not count as correct early. Exact matching makes evaluation deterministic and prevents fuzzy rules from changing results. One-based indexing makes task `t` mean the first `t` tasks, including the current one. |
| 2026-10-19 | Strip surrounding whitespace and trailing Unicode punctuation, but preserve `_`; treat all other non-matches as incorrect. | This accepts harmless sentence punctuation while preserving canonical names such as `credit_limit_change`. No manual correction or lowercasing can introduce evaluator discretion. |
| 2026-10-19 | Load intent-to-domain assignments from the prepared manifest and verify its task order at module import. | This keeps one source of truth for the stage label sets and fails early if the data mapping and evaluation code drift apart. |
| 2026-10-19 | Wrap the fixed prompt body with Qwen's chat template; train loss only on the assistant answer. | The body remains identical across examples while Qwen receives its expected instruction/chat markers. Masking the prompt makes the canonical label, rather than prompt reproduction, the supervised target. |
| 2026-10-19 | Use deterministic greedy decoding with a 16-token generation cap for pilot validation. | The task requires one short label; greedy decoding removes sampling variance and the cap bounds malformed long responses. |
| 2026-10-19 | Use a constant `1e-4` schedule, zero warmup and weight decay, and reset AdamW state at each domain while carrying the adapter. | The fixed schedule preserves the specified learning rate for the one epoch. A new optimizer prevents earlier-task moments from changing the next task's update rule; carrying the adapter preserves sequential LoRA learning. |
| 2026-10-19 | On any 128-token GPU OOM, rerun both domains at 96; on another OOM, rerun both on CPU FP32 at 96. | Restarting the entire pilot avoids comparing domains trained with different settings. FP32 is selected only for CPU fallback because the requested FP16 path is the GPU pilot. Every attempt is logged. |
| 2026-10-19 | Use the observed Trainer global step count rather than forcing a ceiling-rounded update count. | The supplied settings produced 187 updates for each full epoch. Forcing 188 could repeat examples or exceed one epoch, so the measured Trainer behavior is retained and documented. |
| 2026-10-19 | Dynamically pad each batch and fail if a fully formatted example exceeds the sequence cap. | This avoids padding every single-example batch to the full 128 tokens while preserving the utterance and target verbatim. Silent truncation could corrupt either the input or label, so overlength data should stop the run visibly. |
| 2026-10-19 | Record the first pilot as executed before neutral generation-config cleanup. | The model's `temperature`, `top_p`, and `top_k` defaults triggered a warning because greedy decoding was enabled. Those sampling parameters are ignored by greedy decoding, so the recorded predictions remain deterministic; the script now passes neutral values to keep future logs quieter. |
| 2026-10-19 | Replay uses six distinct current examples plus two buffer draws per eight-microbatch accumulation group; current examples are shuffled once per domain. | This preserves one pass over current data while realizing the 6:2 mix. Each replay task has 250 optimizer updates because 1,500 current examples divided into groups of six gives 250 accumulated batches; replay examples are sampled uniformly with replacement from the reservoir. |
| 2026-10-19 | Implement O-LoRA with one frozen adapter per domain, cumulative adapter activation at inference, and squared cross-product orthogonality penalty on LoRA A subspaces (λ=0.5); average each rank-by-rank penalty matrix before summing. | The paper defines task subspaces through columns of A and freezes earlier adapters. Averaging entries keeps penalty scale less dependent on rank; λ follows the paper's reported setting. Keeping all learned adapters active matches cumulative prediction without a domain ID; adapter files remain individually auditable. |
| 2026-10-19 | Evaluate validation and test at every task boundary and retain per-example JSONL plus CSV accuracy matrices. | Per-domain results expose forgetting hidden by a single aggregate. Test data is used only for final reporting and does not select hyperparameters or checkpoints. |
| 2026-10-19 | Match O-LoRA trainable parameters by their full `lora_A.<adapter>` / `lora_B.<adapter>` names; allow running a selected method alone. | Adapter names are nested after the matrix type in PEFT parameter names. The initial substring check did not match that layout, so it was corrected before the O-LoRA phase; independent selection allows rerunning O-LoRA without repeating completed baselines. |
| 2026-10-19 | Use seeded shuffled sampling for sequential LoRA and O-LoRA; preserve ordered accumulation groups only for replay. | The pilot used Trainer shuffling. An initial uncheckpointed attempt used fixed row order, which was not comparable; it was stopped before task 1 completed and no result from it is retained. Replay must preserve its prepared 6:2 group proportions, so its sampler remains ordered. |
| 2026-10-19 | Remove the unused PEFT bootstrap adapter before the first O-LoRA task. | The base wrapper initially creates a `default` adapter; keeping it would add a non-task subspace to the O-LoRA composition and orthogonality penalty. Only the ten learned task adapters should be saved and combined. |
| 2026-10-19 | Activate the O-LoRA adapter stack through PEFT's LoRA tuner rather than the `PeftModel.set_adapter` wrapper. | The pinned PEFT 0.13.2 wrapper accepts one adapter name, while its LoRA tuner supports a list. Calling the tuner API is needed to sum the learned adapters for training/evaluation without a domain ID. |
| 2026-10-19 | Score the untouched base model once with all in-scope labels enabled as the zero-shot reference. | The fixed experiment choices require an initial base score; this is a single reference result, separate from the three continual-learning methods. It uses no OOS examples and cannot affect checkpoint or hyperparameter selection. |
| 2026-10-19 | Save adapter tensors using PEFT's `get_peft_model_state_dict` helper. | Transformers exposes a similarly named accessor that failed on this PEFT-wrapped model. The failed first shuffled epoch completed training but produced no checkpoint or evaluation, so it is discarded and the task restarts with PEFT's adapter-state API. |
| 2026-10-19 | Shuffle replay task 1 like the sequential baseline; use ordered sampling only on replay tasks with the 2,000-row 6:2 schedule. | Task 1 has no old examples, so it should match the ordinary seeded one-epoch procedure. Ordered sampling is required only later to keep each accumulation group at six current plus two replay examples. |
| 2026-10-20 | Resume sequential LoRA from the saved task-boundary adapter after task-8 test generation ended with a CUDA CUBLAS execution error. | Training and validation through task 8 are checkpointed, and test evaluations through task 7 are complete. Restarting task-8 test scoring from its saved adapter avoids retraining tasks 1–8; the partial task-8 prediction file is overwritten so only a complete evaluation is retained. The failure was not an OOM and did not justify changing model, precision, or sequence length. `run_clinc150_methods.py` adds `--resume-sequential-from`; this recovery path still needs to be executed. |
| 2026-10-20 | Preserve and resume partial evaluation JSONL outputs, flushing each prediction and retrying one failed CUDA generation after clearing cached allocations. | The task-10 test evaluation hit CUDA OOM after 1,965 predictions despite a complete saved checkpoint. Durable per-row output prevents repeated successful inference, and a one-time retry can recover transient allocation failures without changing prompts, decoding, model, or precision. The task-10 continuation should resume from the task-10 checkpoint and verified partial prefix. |
| 2026-10-20 | Resume replay from the completed task-8 adapter and saved reservoir, then continue into O-LoRA. | The process stopped after replay task 8 completed and before task 9 began training. Restarting at task 1 would waste completed work; resuming from the exact checkpoint and reservoir preserves the original task sequence and replay sample history. `run_clinc150_methods.py` adds `--resume-replay-from`; `README.md` records the stopped state and continuation choice. |
| 2026-10-20 | Upcast dynamically added O-LoRA adapter weights to FP32 before training. | PEFT's initial adapter constructor upcasts adapter weights on an FP16 base, but `PeftModel.add_adapter` does not. FP16 gradients triggered GradScaler's `Attempting to unscale FP16 gradients` error on the first O-LoRA optimizer update. Casting only the newly added LoRA A/B parameters to FP32 matches PEFT's initial adapter behavior and leaves the frozen base in FP16; O-LoRA is restarted from task 1 because no adapter checkpoint had been saved. |
| 2026-10-21 | Complete O-LoRA from task 1 after the FP32 adapter fix, then record all three method summaries. | The corrected run completed all ten training and evaluation stages without another gradient-scaling failure. The full result set is now available in the per-method CSV/JSONL files, summaries, and adapter storage report; no further experiment run is pending. |
| 2026-10-21 | Report per-domain stage accuracy matrices alongside per-domain forgetting and backward transfer. | Final aggregate accuracy alone hides which earlier domains were retained. The three existing test accuracy matrices provide each stage-by-domain score; the README links them directly and summarizes immediate-to-final accuracy, forgetting, and backward transfer for every domain. |

When a later step forces a new choice, append it here before or alongside the implementation, explain why it was needed, and note the files it affects.


