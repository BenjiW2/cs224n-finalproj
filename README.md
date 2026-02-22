# CS224N Final Project: Instruction -> Action Program Modeling

This repo trains and evaluates a small Toolformer-style model that maps natural language instructions to a structured action program.

Example:

- Instruction: `turn left a bit, then move forward for a while, then bark`
- Program: `[left:30][forward:60][bark:0]`

The codebase includes:

- synthetic data generation
- challenge split construction (IID, length generalization, held-out compositions)
- supervised fine-tuning (SFT) on GPT-2
- evaluation with strict and trajectory-based metrics
- a pseudo-label/self-training step

## Repository Layout

- `src/actions.py`: DSL definition, validation/parsing, serialization.
- `src/data_gen.py`: synthetic instruction-program pair generation.
- `src/splits.py`: split creation and distribution stats.
- `src/train_sft.py`: SFT training with Hugging Face `Trainer`.
- `src/eval.py`: decoding + evaluation metrics.
- `src/sim.py`: simple 2D simulator and trajectory similarity.
- `src/self_train.py`: candidate generation + pseudo-label filtering.
- `src/utils.py`: model/tokenizer loading and constrained decoding FSM.
- `src/toolformer.py`: prototype/scratch file (not main training pipeline).

## Environment Setup

From repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch transformers accelerate
```

Recommended Python: 3.10+.

## Data Format

JSONL rows (used by training/eval) look like:

```json
{"instruction":"go forward a little, then bark","program":"[forward:30][bark:0]","actions":[["forward",30],["bark",0]],"length":2,"template_id":"linear"}
```

Training/eval scripts rely on `instruction` and `program`. Other fields are used for split analysis.

## End-to-End Pipeline

### 1) Generate synthetic data

```bash
mkdir -p data outputs
python3 -m src.data_gen \
  --out data/synth.jsonl \
  --n_programs 6000 \
  --k_paraphrases 3 \
  --min_len 1 \
  --max_len 4 \
  --seed 0
```

### 2) Create splits

```bash
python3 -m src.splits \
  --infile data/synth.jsonl \
  --outdir data \
  --seed 0 \
  --iid_train 14000 \
  --iid_dev 2000 \
  --iid_test 2000
```

This writes:

- `data/iid_train.jsonl`, `data/iid_dev.jsonl`, `data/iid_test.jsonl`
- `data/len_train.jsonl`, `data/len_test.jsonl`
- `data/held_train.jsonl`, `data/held_test.jsonl`, `data/held_control.jsonl`

### 3) Train base SFT model

```bash
python3 -m src.train_sft \
  --model gpt2 \
  --train data/iid_train.jsonl \
  --dev data/iid_dev.jsonl \
  --out outputs/sft_base \
  --epochs 1 \
  --lr 5e-5 \
  --bsz 2 \
  --grad_accum 16 \
  --max_length 128
```

Optional: add `--fp16` on compatible hardware.

### 4) Evaluate

Greedy constrained decode on IID test:

```bash
python3 -m src.eval \
  --model outputs/sft_base \
  --test data/iid_test.jsonl \
  --constrained 1 \
  --temperature 0.0 \
  --max_new_tokens 64
```

Challenge sets:

```bash
python3 -m src.eval --model outputs/sft_base --test data/len_test.jsonl --constrained 1
python3 -m src.eval --model outputs/sft_base --test data/held_test.jsonl --constrained 1
python3 -m src.eval --model outputs/sft_base --test data/held_control.jsonl --constrained 1
```

## Self-Training (Pseudo Labels)

`src/self_train.py` samples `M` candidate programs per instruction, scores each candidate against reference trajectories, and keeps high-utility pseudo labels (threshold `tau`).

Current script uses hardcoded defaults and is run as:

```bash
python3 src/self_train.py
```

Defaults expect:

- base checkpoint: `outputs/sft_base`
- unlabeled file: `data/unlabeled_len234.jsonl`
- output pseudo labels: `data/pseudo_iter1.jsonl`

## Metrics Reported (`src/eval.py`)

- `valid_rate`: fraction of syntactically valid programs.
- `exact_match`: exact canonical string match vs gold program.
- `step_precision`, `step_recall`, `step_f1`: positional step match on `(tool, value)`.
- `tool_edit_dist`: Levenshtein distance on tool sequences.
- `length_acc`: exact action-count match.
- `mean_traj_score`: simulator trajectory similarity in `[0,1]`.

## Notes / Current Limitations

- `src/toolformer.py` is experimental and not wired into the main pipeline.
- `src/self_train.py` currently has no CLI args.
- `src/actions.py` defines `first_invalid_reason` twice; the second definition overrides the first.
