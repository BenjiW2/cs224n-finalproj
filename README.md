# CS224N Final Project: Instruction -> Action Program Modeling

This repo trains and evaluates a model that maps natural language instructions to a structured action program.

Example:

- Instruction: `turn left a bit, then move forward for a while, then bark`
- Program: `[left:45][forward:60][bark:0]`

The codebase includes:

- synthetic data generation
- challenge split construction (IID, length generalization, held-out compositions)
- supervised fine-tuning (SFT) on GPT-2
- evaluation with strict syntax and sequence-level metrics

## Action Semantics

The action DSL is in `src/actions.py`.

- `[forward:v]`: move forward by distance bin `v`
- `[backward:v]`: move backward by distance bin `v`
- `[left:v]`: turn left by angle bin `v`
- `[right:v]`: turn right by angle bin `v`
- `[bark:0]`: no-op on position/heading, event token only

Bin values:

- Distance bins (`forward/backward`) are `{10, 30, 60, 100}`
- Turn-angle bins (`left/right`) are `{15, 45, 90, 180}`

## Repository Layout

- `src/actions.py`: DSL definition, validation/parsing, serialization.
- `src/data_gen.py`: synthetic instruction-program pair generation.
- `src/splits.py`: split creation and distribution stats.
- `src/train_sft.py`: SFT training with Hugging Face `Trainer`.
- `src/eval.py`: decoding + evaluation metrics.
<<<<<<< HEAD
- `src/self_train.py`: candidate generation + pseudo-label filtering.
=======
>>>>>>> 2bbbbf042721049016cb4f386fffde61580d833a
- `src/utils.py`: model/tokenizer loading and constrained decoding FSM.
- `src/score_predictions.py`: model-agnostic evaluator for predicted programs.
- `src/run_milestone_eval.py`: run a model/split/shot evaluation matrix and save JSONL.

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

Note: if you generated data before the angle-bin update, regenerate it. New valid turn values are `{15,45,90,180}`.

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
  --include_task_spec 0 \
  --max_new_tokens 64
```

Challenge sets:

```bash
python3 -m src.eval --model outputs/sft_base --test data/len_test.jsonl --constrained 1 --include_task_spec 0
python3 -m src.eval --model outputs/sft_base --test data/held_test.jsonl --constrained 1 --include_task_spec 0
python3 -m src.eval --model outputs/sft_base --test data/held_control.jsonl --constrained 1 --include_task_spec 0
```

Few-shot baseline (in-context demos from a train file):

```bash
python3 -m src.eval \
  --model gpt2 \
  --test data/iid_test.jsonl \
  --constrained 0 \
  --temperature 0.0 \
  --fewshot_path data/iid_train.jsonl \
  --num_shots 4 \
  --include_task_spec 1 \
  --fewshot_seed 0
```

Run a full milestone matrix:

```bash
python3 -m src.run_milestone_eval \
  --models gpt2 \
  --tests data/iid_test.jsonl,data/len_test.jsonl,data/held_test.jsonl,data/held_control.jsonl \
  --fewshot_path data/iid_train.jsonl \
  --num_shots_list 0,4 \
  --constrained 1 \
  --include_task_spec 1 \
  --temperature 0.0 \
  --out outputs/milestone_eval.jsonl
```

Qwen pretrained sweep example:

```bash
python3 -m src.run_milestone_eval \
  --models Qwen/Qwen3-0.6B,Qwen/Qwen3-1.7B,Qwen/Qwen3-4B \
  --tests data/iid_test.jsonl,data/len_test.jsonl,data/held_test.jsonl,data/held_control.jsonl \
  --fewshot_path data/iid_train.jsonl \
  --num_shots_list 0,4 \
  --constrained 1 \
  --include_task_spec 1 \
  --temperature 0.0 \
  --out outputs/qwen_pretrained_matrix.jsonl
```

Prompt note:

- `--include_task_spec 1` prepends explicit format instructions (recommended for pretrained zero/few-shot baselines).
- `--include_task_spec 0` uses only `Instruction: ... / Action sequence:` (recommended for SFT checkpoints trained with that minimal prompt format).

## Milestone Report Guide

Use this section as the source of truth when writing the milestone paper.

### Research Questions + Hypotheses

1. Can pretrained LMs produce valid action programs zero-shot?
   Hypothesis: syntax validity is moderate, exact sequence match is low.
2. Does few-shot prompting improve tool-sequence generation?
   Hypothesis: `tool_step_f1` improves more than strict `step_f1`.
3. Can SFT solve IID mapping?
   Hypothesis: IID `exact_match` and `step_f1` increase substantially over pretrained baselines.
4. Does performance drop under distribution shift (length/composition)?
   Hypothesis: `len_test` and `held_test` underperform IID.
5. Is compositional failure specific (held-out combos) rather than general hardness?
   Hypothesis: `held_test` < `held_control` on `tool_step_f1`.

### Canonical Experiment Matrix

Run and report these rows (minimum milestone set):

| Model | Train Split | Test Split | Shots | Purpose |
|---|---|---|---:|---|
| Qwen3-0.6B | none | all 4 tests | 0 | zero-shot baseline |
| Qwen3-1.7B | none | all 4 tests | 0 | zero-shot baseline |
| Qwen3-4B | none | all 4 tests | 0 | zero-shot baseline |
| Qwen3-0.6B | none | all 4 tests | 4 | few-shot baseline |
| Qwen3-1.7B | none | all 4 tests | 4 | few-shot baseline |
| Qwen3-4B | none | all 4 tests | 4 | few-shot baseline |
| Chosen SFT model | `iid_train` | `iid_test` | 0 | IID performance |
| Chosen SFT model | `iid_train` | `len_test` | 0 | length generalization |
| Chosen SFT model | `iid_train` | `held_test` | 0 | compositional generalization |
| Chosen SFT model | `iid_train` | `held_control` | 0 | compositional control |

Optional extensions:

- SFT on `held_train` and evaluate on `held_test`/`held_control`
- SFT on `len_train` and evaluate on `len_test`

### Metrics To Emphasize In The Paper

Primary (table/main claims):

- `exact_match`
- `step_f1` (tool + value)
- `tool_step_f1` (tool identity only)
- `valid_rate`

Secondary (diagnostics):

- `parseable_rate`
- `tool_exact_match`
- `tool_edit_dist`
- `length_acc`
- `invalid_reasons`

### Reproducibility Defaults

Use the following fixed settings unless explicitly ablated:

- Decoding: greedy (`--temperature 0.0`)
- Constrained decoding: on (`--constrained 1`) for main numbers
- Pretrained baselines: `--include_task_spec 1`
- SFT checkpoints: `--include_task_spec 0`
- Few-shot seed: `--fewshot_seed 0`
- Data generation seed: `--seed 0`
- Split seed: `--seed 0`
- Report model names exactly as passed in CLI
- Record hardware (GPU type, VRAM) and runtime per experiment block

### Result Logging Convention

Recommended output files:

- `outputs/pretrained_matrix.jsonl` (Qwen zero/few-shot)
- `outputs/sft_iid_matrix.jsonl`
- `outputs/sft_held_matrix.jsonl` (optional)
- `outputs/sft_len_matrix.jsonl` (optional)

Each JSONL row should represent one `(model, test_split, num_shots)` setting.
Keep all raw JSONL outputs for appendix/reproducibility.

### Error Analysis Protocol

For each key condition (at least pretrained-best and SFT model on `held_test`):

1. Sample 25-50 failures.
2. Label failure type:
   - invalid syntax
   - wrong tool identity
   - correct tools but wrong values
   - wrong sequence length
   - near-miss sequence (high tool overlap, low exact match)
3. Report 3-5 representative examples with:
   - instruction
   - gold program
   - predicted program
   - brief diagnosis

### Milestone Writing Checklist

- Problem statement and task definition are clear.
- Action semantics are stated (`left/right` are turns).
- Data generation and split construction are documented.
- Baselines include pretrained zero/few-shot.
- Main table includes IID, length shift, held-out composition, and control.
- Metrics include both strict and tool-only views.
- Error analysis includes qualitative examples + category counts.
- Limitations and next steps are explicit.

## Scoring Predictions Directly (No Model Dependency)

If you already have predicted programs (or want quick baseline numbers), evaluate them directly:

```bash
python3 -m src.score_predictions \
  --gold data/iid_test.jsonl \
  --pred preds.jsonl \
  --pred_key pred_program
```

`--pred` can be:

- JSONL with `pred_program` (or `program`) field per row
- plain text file with one predicted program per line

## Metrics Reported (`src/eval.py`)

- `valid_rate`: fraction of syntactically valid programs.
- `parseable_rate`: fraction that are parseable with loose parser.
- `exact_match`: exact canonical string match vs gold program.
- `tool_exact_match`: exact tool sequence match ignoring values.
- `step_precision`, `step_recall`, `step_f1`: positional step match on `(tool, value)`.
- `tool_step_precision`, `tool_step_recall`, `tool_step_f1`: positional step match on tool identity only.
- `tool_edit_dist`: Levenshtein distance on tool sequences.
- `length_acc`: exact action-count match.

## Notes / Current Limitations

- `src/actions.py` defines `first_invalid_reason` twice; the second definition overrides the first.
