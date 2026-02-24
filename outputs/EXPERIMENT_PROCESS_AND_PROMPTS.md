# Experiment Process And Prompts

## What was run
- Models: Qwen/Qwen3-0.6B, Qwen/Qwen3-1.7B, Qwen/Qwen3-4B
- Shot settings: [0, 4, 16]
- Test splits: held_control_20, held_test_20, iid_test_20, len_test_20
- Rows: 720 (expected 3 models x 3 shot settings x 4 splits x 20 examples = 720)
- Decoding: constrained=1, temperature=0.0, max_new_tokens=32, fewshot_strategy=diverse, include_task_spec=1
- Few-shot source used to reconstruct prompts: `data/iid_train_tiny.jsonl` (seed=0, strategy=diverse)

## Results

All numbers below come from the 20-example split runs (3 models x 3 shot settings x 4 splits) plus the Qwen3-0.6B SFT run.

### Pretrained Summary (Averaged Across 4 Splits)

| Model | Shots | Avg Exact Match | Avg Tool-Step F1 | Avg Length Accuracy | Avg Strict Valid |
|---|---:|---:|---:|---:|---:|
| Qwen/Qwen3-0.6B | 0 | 0.00% | 0.193 | 6.25% | 100.00% |
| Qwen/Qwen3-0.6B | 4 | 1.25% | 0.400 | 6.25% | 100.00% |
| Qwen/Qwen3-0.6B | 16 | 1.25% | 0.400 | 6.25% | 100.00% |
| Qwen/Qwen3-1.7B | 0 | 1.25% | 0.224 | 6.25% | 100.00% |
| Qwen/Qwen3-1.7B | 4 | 1.25% | 0.400 | 6.25% | 100.00% |
| Qwen/Qwen3-1.7B | 16 | 1.25% | 0.400 | 6.25% | 100.00% |
| Qwen/Qwen3-4B | 0 | 0.00% | 0.400 | 6.25% | 100.00% |
| Qwen/Qwen3-4B | 4 | 1.25% | 0.369 | 10.00% | 100.00% |
| Qwen/Qwen3-4B | 16 | 0.00% | 0.436 | 12.50% | 100.00% |

### Pretrained Exact Matches By Split (Count / 20)

| Model | Shots | IID | Length Gen | Held-Out | Held Control |
|---|---:|---:|---:|---:|---:|
| Qwen/Qwen3-0.6B | 0 | 0/20 | 0/20 | 0/20 | 0/20 |
| Qwen/Qwen3-0.6B | 4 | 1/20 | 0/20 | 0/20 | 0/20 |
| Qwen/Qwen3-0.6B | 16 | 1/20 | 0/20 | 0/20 | 0/20 |
| Qwen/Qwen3-1.7B | 0 | 1/20 | 0/20 | 0/20 | 0/20 |
| Qwen/Qwen3-1.7B | 4 | 1/20 | 0/20 | 0/20 | 0/20 |
| Qwen/Qwen3-1.7B | 16 | 1/20 | 0/20 | 0/20 | 0/20 |
| Qwen/Qwen3-4B | 0 | 0/20 | 0/20 | 0/20 | 0/20 |
| Qwen/Qwen3-4B | 4 | 1/20 | 0/20 | 0/20 | 0/20 |
| Qwen/Qwen3-4B | 16 | 0/20 | 0/20 | 0/20 | 0/20 |

### Qwen3-0.6B SFT (IID-trained)

| Split | Exact Match | Strict Valid |
|---|---:|---:|
| iid_test_20 | 3/20 (15.00%) | 20/20 (100.00%) |
| len_test_20 | 0/20 (0.00%) | 20/20 (100.00%) |
| held_test_20 | 0/20 (0.00%) | 20/20 (100.00%) |
| held_control_20 | 0/20 (0.00%) | 20/20 (100.00%) |

Overall SFT (all 80 rows): exact = 3/80 (3.75%), strict valid = 80/80 (100.00%).

### Qwen3-0.6B: Pretrained 0-Shot vs SFT (Exact Match)

| Split | Pretrained 0-shot | SFT | Delta (SFT - Pre) |
|---|---:|---:|---:|
| iid_test_20 | 0.00% | 15.00% | +15.00 pp |
| len_test_20 | 0.00% | 0.00% | +0.00 pp |
| held_test_20 | 0.00% | 0.00% | +0.00 pp |
| held_control_20 | 0.00% | 0.00% | +0.00 pp |

Notes:
- `strict_valid` is near-perfect, so most failures are semantic (wrong tool/value/length), not syntax.
- On this 20-row setting, 4-shot/16-shot improved tool-level alignment much more than strict exact-match.
- SFT improved IID exact-match for Qwen3-0.6B (0% -> 15%), but did not improve OOD exact-match on length/compositional splits.

## Prompt templates from code
`src/eval.py` constants:

```text
PROMPT_TMPL = "Instruction: {instr}\nAction sequence:"
DEMO_TMPL = "Instruction: {instr}\nAction sequence: {prog}"

You are converting instructions into an action sequence program.
Output only the program, with no extra text.
Program format: concatenate bracketed calls with no separators.
Each call format: [tool:value]
Allowed tools: forward, backward, left, right, bark
Allowed values for forward/backward: 10, 30, 60, 100
Allowed values for left/right: 15, 45, 90, 180
Allowed value for bark: 0
Example format only: [left:45][forward:60][bark:0]
```

## Full few-shot demo block (4-shot)

```text
Instruction: okay rotate left about a right angle, after that um let out a bark, then okay move back a tiny bit, next um move ahead a bit.
Action sequence: [left:90][bark:0][backward:10][forward:30]

Instruction: carefully rotate left a lot, next advance some distance, and then kind of rotate right slightly, next veer left a bit
Action sequence: [left:90][forward:60][right:15][left:45]

Instruction: um veer right slightly, then please turn right about-face, after that please move ahead for a long while
Action sequence: [right:15][right:180][forward:100]

Instruction: First, veer left slightly. Then, advance a bit. After that, advance just a little. Next, rotate right about a quarter turn.
Action sequence: [left:15][forward:30][forward:10][right:45]
```

## Full few-shot demo block (16-shot)

```text
Instruction: okay rotate left about a right angle, after that um let out a bark, then okay move back a tiny bit, next um move ahead a bit.
Action sequence: [left:90][bark:0][backward:10][forward:30]

Instruction: carefully rotate left a lot, next advance some distance, and then kind of rotate right slightly, next veer left a bit
Action sequence: [left:90][forward:60][right:15][left:45]

Instruction: um veer right slightly, then please turn right about-face, after that please move ahead for a long while
Action sequence: [right:15][right:180][forward:100]

Instruction: First, veer left slightly. Then, advance a bit. After that, advance just a little. Next, rotate right about a quarter turn.
Action sequence: [left:15][forward:30][forward:10][right:45]

Instruction: First, back up for a while. Then, bark. After that, rotate left a lot. Next, rotate right about a quarter turn.
Action sequence: [backward:60][bark:0][left:90][right:45]

Instruction: First, move ahead slightly. Then, veer left a bit. After that, go backward for a long while.
Action sequence: [forward:10][left:45][backward:100]

Instruction: First, veer right about-face. Then, bark. After that, turn right a tiny bit. Next, move back a bit.
Action sequence: [right:180][bark:0][right:15][backward:30]

Instruction: rotate right about-face, then back up for a long while, after that move ahead a lot
Action sequence: [right:180][backward:100][forward:100]

Instruction: um veer right a lot, and then veer left a bit, next okay veer right a tiny bit.
Action sequence: [right:90][left:45][right:15]

Instruction: okay move ahead slightly, then please reverse a tiny bit, then kind of turn left a lot
Action sequence: [forward:10][backward:10][left:90]

Instruction: First, rotate right a lot. Then, move back a tiny bit. After that, let out a bark.
Action sequence: [right:90][backward:10][bark:0]

Instruction: veer right a lot, after that um move back for a long while, after that kind of back up just a little.
Action sequence: [right:90][backward:100][backward:10]

Instruction: kind of move forward a bit, next kind of move forward for a long while, after that kind of rotate right slightly.
Action sequence: [forward:30][forward:100][right:15]

Instruction: kind of go forward slightly, then okay veer right slightly.
Action sequence: [forward:10][right:15]

Instruction: First, veer right a bit. Then, go forward for a while. After that, rotate right about a quarter turn.
Action sequence: [right:45][forward:60][right:45]

Instruction: First, make a bark sound. Then, rotate right a bit.
Action sequence: [bark:0][right:45]
```

## Example fully rendered prompts (same instruction)

Instruction used: `First, bark. Then, advance some distance. After that, move ahead slightly.`

### 0-shot

```text
You are converting instructions into an action sequence program.
Output only the program, with no extra text.
Program format: concatenate bracketed calls with no separators.
Each call format: [tool:value]
Allowed tools: forward, backward, left, right, bark
Allowed values for forward/backward: 10, 30, 60, 100
Allowed values for left/right: 15, 45, 90, 180
Allowed value for bark: 0
Example format only: [left:45][forward:60][bark:0]

Instruction: First, bark. Then, advance some distance. After that, move ahead slightly.
Action sequence:
```

### 4-shot

```text
You are converting instructions into an action sequence program.
Output only the program, with no extra text.
Program format: concatenate bracketed calls with no separators.
Each call format: [tool:value]
Allowed tools: forward, backward, left, right, bark
Allowed values for forward/backward: 10, 30, 60, 100
Allowed values for left/right: 15, 45, 90, 180
Allowed value for bark: 0
Example format only: [left:45][forward:60][bark:0]

Instruction: okay rotate left about a right angle, after that um let out a bark, then okay move back a tiny bit, next um move ahead a bit.
Action sequence: [left:90][bark:0][backward:10][forward:30]

Instruction: carefully rotate left a lot, next advance some distance, and then kind of rotate right slightly, next veer left a bit
Action sequence: [left:90][forward:60][right:15][left:45]

Instruction: um veer right slightly, then please turn right about-face, after that please move ahead for a long while
Action sequence: [right:15][right:180][forward:100]

Instruction: First, veer left slightly. Then, advance a bit. After that, advance just a little. Next, rotate right about a quarter turn.
Action sequence: [left:15][forward:30][forward:10][right:45]

Instruction: First, bark. Then, advance some distance. After that, move ahead slightly.
Action sequence:
```

### 16-shot

```text
You are converting instructions into an action sequence program.
Output only the program, with no extra text.
Program format: concatenate bracketed calls with no separators.
Each call format: [tool:value]
Allowed tools: forward, backward, left, right, bark
Allowed values for forward/backward: 10, 30, 60, 100
Allowed values for left/right: 15, 45, 90, 180
Allowed value for bark: 0
Example format only: [left:45][forward:60][bark:0]

Instruction: okay rotate left about a right angle, after that um let out a bark, then okay move back a tiny bit, next um move ahead a bit.
Action sequence: [left:90][bark:0][backward:10][forward:30]

Instruction: carefully rotate left a lot, next advance some distance, and then kind of rotate right slightly, next veer left a bit
Action sequence: [left:90][forward:60][right:15][left:45]

Instruction: um veer right slightly, then please turn right about-face, after that please move ahead for a long while
Action sequence: [right:15][right:180][forward:100]

Instruction: First, veer left slightly. Then, advance a bit. After that, advance just a little. Next, rotate right about a quarter turn.
Action sequence: [left:15][forward:30][forward:10][right:45]

Instruction: First, back up for a while. Then, bark. After that, rotate left a lot. Next, rotate right about a quarter turn.
Action sequence: [backward:60][bark:0][left:90][right:45]

Instruction: First, move ahead slightly. Then, veer left a bit. After that, go backward for a long while.
Action sequence: [forward:10][left:45][backward:100]

Instruction: First, veer right about-face. Then, bark. After that, turn right a tiny bit. Next, move back a bit.
Action sequence: [right:180][bark:0][right:15][backward:30]

Instruction: rotate right about-face, then back up for a long while, after that move ahead a lot
Action sequence: [right:180][backward:100][forward:100]

Instruction: um veer right a lot, and then veer left a bit, next okay veer right a tiny bit.
Action sequence: [right:90][left:45][right:15]

Instruction: okay move ahead slightly, then please reverse a tiny bit, then kind of turn left a lot
Action sequence: [forward:10][backward:10][left:90]

Instruction: First, rotate right a lot. Then, move back a tiny bit. After that, let out a bark.
Action sequence: [right:90][backward:10][bark:0]

Instruction: veer right a lot, after that um move back for a long while, after that kind of back up just a little.
Action sequence: [right:90][backward:100][backward:10]

Instruction: kind of move forward a bit, next kind of move forward for a long while, after that kind of rotate right slightly.
Action sequence: [forward:30][forward:100][right:15]

Instruction: kind of go forward slightly, then okay veer right slightly.
Action sequence: [forward:10][right:15]

Instruction: First, veer right a bit. Then, go forward for a while. After that, rotate right about a quarter turn.
Action sequence: [right:45][forward:60][right:45]

Instruction: First, make a bark sound. Then, rotate right a bit.
Action sequence: [bark:0][right:45]

Instruction: First, bark. Then, advance some distance. After that, move ahead slightly.
Action sequence:
```


## File map
- Per-example readable CSV with prompts: `outputs/qwen3_3models_0_4_16_20rows_with_prompts.csv`
- Main process + prompts + results doc: `outputs/EXPERIMENT_PROCESS_AND_PROMPTS.md`
- Pretrained metric summaries: `modal_outputs/outputs/qwen3_06b_20.jsonl`, `modal_outputs/outputs/qwen3_17b_20.jsonl`, `modal_outputs/outputs/qwen3_4b_20.jsonl`
- Pretrained per-example raw outputs: `modal_outputs/outputs/qwen3_raw_*_20_*.jsonl`
- SFT per-example readable dump (with split labels): `outputs/qwen06b_sft_20.txt`
- SFT compact CSV: `outputs/qwen06b_sft_20_min.csv`
