# CLINC150 Continual LoRA

A single-device continual-learning comparison on CLINC150 using `Qwen/Qwen2.5-0.5B-Instruct` and a frozen base model. It compares sequential LoRA, replay LoRA, and O-LoRA over the fixed order `auto_and_commute → banking → credit_cards → home → kitchen_and_dining → meta → small_talk → travel → utility → work`. The OOS examples are excluded.

All methods use LoRA rank 4, alpha 8, dropout 0.05, `q_proj`/`v_proj` targets, FP16, sequence length 128, one epoch per domain, batch size 1 with accumulation 8, and learning rate `1e-4`. Replay uses a 300-example reservoir with a 6:2 current-to-replay mix. O-LoRA uses one adapter per domain with orthogonality coefficient `λ=0.5`.

## Results

| Method | Final test accuracy | Mean forgetting | Backward transfer |
|---|---:|---:|---:|
| Sequential LoRA | 25.76% | 60.78 pp | −60.78 pp |
| Replay LoRA | 57.07% | 29.44 pp | −29.13 pp |
| O-LoRA | 28.91% | 55.07 pp | −55.07 pp |

The test accuracy matrices show per-domain accuracy after each task: [sequential](results/sequential/test_accuracy_matrix.csv), [replay](results/replay/test_accuracy_matrix.csv), and [O-LoRA](results/olora/test_accuracy_matrix.csv). Each method's `final_summary.json` reports forgetting and backward transfer; `test_metrics.csv` records cumulative accuracy, invalid outputs, and per-stage totals. Training time, validation metrics, GPU snapshots, and pinned environment versions are also in `results/`.

These results describe this 0.5B model on one GTX 1650 with 4 GB VRAM. They do not establish how 4B or 7B models would perform. The official CLINC150 train/validation/test partitions were preserved, but the source contains a duplicate `(utterance, intent)` row across train and validation (`hey what's up`, `greeting`); see the [dataset manifest](data/manifest.json) and [decision log](DECISIONS.md) for this limitation and its handling.

## Setup and run

Use Python 3.11 on Windows. Install the CUDA-enabled PyTorch 2.5.1 wheel first, then the pinned libraries:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python .\verify_setup.py
```

The repository includes the prepared in-scope splits and manifest, so run the experiment directly:

```powershell
python .\run_clinc150_methods.py --zero-shot-only
python .\run_clinc150_methods.py
```

The data manifest records the source URL and checksum, split files, intent/domain mapping, counts, and seed. To regenerate the prepared splits, download the UCI full archive and place `clinc150-uci.zip` under `data/raw/` and its `clinc150_uci/data_full.json` at `data/source/clinc150_uci/data_full.json`, then run `python .\prepare_clinc150.py`. The UCI license and metadata are included under `data/source/clinc150_uci/`; the duplicate raw archive and source JSON are excluded from Git.

## What to publish

The compact `results/` bundle contains the summaries and accuracy matrices needed to review the reported findings. Full per-example generations and intermediate run logs are kept locally under `run-logs/` and are excluded from Git by `.gitignore`; they are not required for the code or compact results. Adapter checkpoints under `artifacts/` are also excluded by default and can be added separately if you want to distribute the trained adapters.

`DECISIONS.md` preserves the detailed experiment specification, implementation choices, rationale, and recovery history.
