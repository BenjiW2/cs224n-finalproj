# Pretrained and SFT Runs From Today: Extra Evaluation Analysis

## Scope

This document covers both categories of runs produced today on the new `final18k` dataset:

1. Pretrained non-SFT runs from `finalproj_outputs/outputs/final_raw/pre_qwen*.jsonl`.
2. The new IID SFT run from `finalproj_outputs/outputs/qwen06b_sft_iid_18k/` and its saved raw eval outputs.

Pretrained suite:
- Models: `Qwen3-0.6B`, `Qwen3-1.7B`, `Qwen3-4B`
- Shot settings: `0`, `4`, `16`
- Splits: `iid_test`, `len_test`, `held_test`, `held_control`
- Total pretrained evaluations: `36`

SFT suite currently available in this document:
- Model: `Qwen/Qwen3-0.6B`
- Train split: `data/final18k/iid_train.jsonl` with `12000` examples
- Dev split: `data/final18k/iid_dev.jsonl` with `1500` examples
- Epochs: `3.0`
- Effective batch size: `32`
- Estimated optimizer steps: `1125`
- Prompt includes task spec: `True`
- LoRA enabled: `True`

Dataset sizes for the evaluated `final18k` splits:
- `iid_test`: `1500`
- `len_test`: `1100`
- `held_test`: `601`
- `held_control`: `365`

## Metric Definitions

- `exact_match`: full sequence match including magnitudes.
- `exact_match_ignore_magnitude`: full sequence match on tool identities only; magnitudes are ignored.
- `magnitude_gap_exact`: `exact_match_ignore_magnitude - exact_match`; this isolates the magnitude penalty for exact sequence match.
- `step_f1`: positional step F1 including magnitudes.
- `step_f1_ignore_magnitude`: positional step F1 on tool identities only.
- `magnitude_gap_step_f1`: `step_f1_ignore_magnitude - step_f1`; this isolates the magnitude penalty at the step level.
- `first_call_acc`: first predicted action must match the first gold action exactly, including magnitude.
- `first_tool_acc`: first predicted tool must match the first gold tool, ignoring magnitude.
- `length_acc`: predicted program length matches gold length.
- `tool_edit_dist`: edit distance on tool identities only.

## Pretrained Executive Summary

1. Strict exact match is extremely low everywhere. The best single strict result in the entire pretrained suite is `0.0725` on `iid_test` from `Qwen3-4B` at `16` shots.
2. Ignoring magnitude helps a lot, but not enough to make the pretrained models look strong. The best ignore-magnitude exact match is `0.2025`, again from `Qwen3-4B` at `16` shots on `iid_test`.
3. The models often get the first tool right while still getting the first call wrong. Average `first_tool_acc` reaches `0.8290` for `Qwen3-4B` at `16` shots, but average `first_call_acc` for the same configuration is only `0.2013`.
4. Few-shot prompting helps much more than model scaling alone on tool-identity metrics. The clearest jump is `0 -> 4` shots.
5. OOD behavior remains poor. `len_test` and `held_test` stay near zero on strict exact match even when first-tool accuracy is fairly high.

## Pretrained Average Over Four Splits

| model | shots | avg_exact | avg_exact_ignore_mag | avg_gap_exact | avg_step_f1 | avg_step_f1_ignore_mag | avg_gap_step_f1 | avg_first_call | avg_first_tool | avg_length_acc | avg_tool_edit_dist |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-0.6B | 0 | 0.0056 | 0.0297 | 0.0241 | 0.0414 | 0.1918 | 0.1504 | 0.0716 | 0.3258 | 0.0681 | 2.3318 |
| Qwen3-0.6B | 4 | 0.0081 | 0.0463 | 0.0382 | 0.1062 | 0.4410 | 0.3348 | 0.1965 | 0.8113 | 0.0638 | 2.0197 |
| Qwen3-0.6B | 16 | 0.0169 | 0.0501 | 0.0332 | 0.1424 | 0.4429 | 0.3005 | 0.2561 | 0.8113 | 0.0676 | 2.0402 |
| Qwen3-1.7B | 0 | 0.0037 | 0.0206 | 0.0169 | 0.0644 | 0.2375 | 0.1731 | 0.1206 | 0.4332 | 0.0600 | 2.3246 |
| Qwen3-1.7B | 4 | 0.0100 | 0.0425 | 0.0325 | 0.1247 | 0.4397 | 0.3150 | 0.2374 | 0.8113 | 0.0600 | 2.0481 |
| Qwen3-1.7B | 16 | 0.0100 | 0.0425 | 0.0325 | 0.1247 | 0.4397 | 0.3150 | 0.2374 | 0.8113 | 0.0600 | 2.0557 |
| Qwen3-4B | 0 | 0.0069 | 0.0425 | 0.0356 | 0.1096 | 0.4397 | 0.3302 | 0.2072 | 0.8113 | 0.0600 | 2.0722 |
| Qwen3-4B | 4 | 0.0081 | 0.0425 | 0.0344 | 0.1039 | 0.4422 | 0.3383 | 0.1923 | 0.8076 | 0.0629 | 2.1193 |
| Qwen3-4B | 16 | 0.0181 | 0.0506 | 0.0325 | 0.1166 | 0.4676 | 0.3511 | 0.2013 | 0.8290 | 0.0742 | 2.2515 |

## Pretrained Split-Level Results

| model | shots | split | exact | exact_ignore_mag | gap_exact | step_f1 | step_f1_ignore_mag | gap_step_f1 | first_call | first_tool | length_acc | tool_edit_dist |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-0.6B | 0 | iid_test | 0.0225 | 0.1100 | 0.0875 | 0.0528 | 0.2416 | 0.1889 | 0.0775 | 0.3500 | 0.2475 | 1.9700 |
| Qwen3-0.6B | 4 | iid_test | 0.0325 | 0.1725 | 0.1400 | 0.1119 | 0.4961 | 0.3842 | 0.1850 | 0.7975 | 0.2425 | 1.6725 |
| Qwen3-0.6B | 16 | iid_test | 0.0675 | 0.1750 | 0.1075 | 0.1705 | 0.4973 | 0.3268 | 0.2625 | 0.7975 | 0.2450 | 1.6925 |
| Qwen3-1.7B | 0 | iid_test | 0.0150 | 0.0825 | 0.0675 | 0.0655 | 0.2642 | 0.1987 | 0.1125 | 0.4275 | 0.2400 | 1.9800 |
| Qwen3-1.7B | 4 | iid_test | 0.0400 | 0.1700 | 0.1300 | 0.1366 | 0.4952 | 0.3587 | 0.2300 | 0.7975 | 0.2400 | 1.6925 |
| Qwen3-1.7B | 16 | iid_test | 0.0400 | 0.1700 | 0.1300 | 0.1366 | 0.4952 | 0.3587 | 0.2300 | 0.7975 | 0.2400 | 1.6975 |
| Qwen3-4B | 0 | iid_test | 0.0275 | 0.1700 | 0.1425 | 0.1132 | 0.4952 | 0.3821 | 0.1925 | 0.7975 | 0.2400 | 1.7100 |
| Qwen3-4B | 4 | iid_test | 0.0325 | 0.1700 | 0.1375 | 0.1127 | 0.4977 | 0.3850 | 0.1850 | 0.7950 | 0.2425 | 1.7525 |
| Qwen3-4B | 16 | iid_test | 0.0725 | 0.2025 | 0.1300 | 0.1533 | 0.5428 | 0.3895 | 0.2225 | 0.8375 | 0.2500 | 1.8250 |
| Qwen3-0.6B | 0 | len_test | 0.0000 | 0.0033 | 0.0033 | 0.0398 | 0.1765 | 0.1367 | 0.0724 | 0.3191 | 0.0099 | 2.4178 |
| Qwen3-0.6B | 4 | len_test | 0.0000 | 0.0033 | 0.0033 | 0.1045 | 0.4291 | 0.3246 | 0.2007 | 0.8257 | 0.0033 | 2.1086 |
| Qwen3-0.6B | 16 | len_test | 0.0000 | 0.0066 | 0.0066 | 0.1355 | 0.4307 | 0.2952 | 0.2566 | 0.8257 | 0.0066 | 2.1349 |
| Qwen3-1.7B | 0 | len_test | 0.0000 | 0.0000 | 0.0000 | 0.0664 | 0.2390 | 0.1726 | 0.1283 | 0.4539 | 0.0000 | 2.3980 |
| Qwen3-1.7B | 4 | len_test | 0.0000 | 0.0000 | 0.0000 | 0.1271 | 0.4280 | 0.3009 | 0.2500 | 0.8257 | 0.0000 | 2.1349 |
| Qwen3-1.7B | 16 | len_test | 0.0000 | 0.0000 | 0.0000 | 0.1271 | 0.4280 | 0.3009 | 0.2500 | 0.8257 | 0.0000 | 2.1414 |
| Qwen3-4B | 0 | len_test | 0.0000 | 0.0000 | 0.0000 | 0.1127 | 0.4280 | 0.3152 | 0.2171 | 0.8257 | 0.0000 | 2.1579 |
| Qwen3-4B | 4 | len_test | 0.0000 | 0.0000 | 0.0000 | 0.1056 | 0.4312 | 0.3257 | 0.2007 | 0.8224 | 0.0033 | 2.2138 |
| Qwen3-4B | 16 | len_test | 0.0000 | 0.0000 | 0.0000 | 0.1063 | 0.4477 | 0.3414 | 0.1974 | 0.8355 | 0.0132 | 2.3520 |
| Qwen3-0.6B | 0 | held_test | 0.0000 | 0.0056 | 0.0056 | 0.0439 | 0.1848 | 0.1409 | 0.0899 | 0.3539 | 0.0056 | 2.5000 |
| Qwen3-0.6B | 4 | held_test | 0.0000 | 0.0000 | 0.0000 | 0.1052 | 0.4414 | 0.3361 | 0.2135 | 0.8933 | 0.0000 | 2.2697 |
| Qwen3-0.6B | 16 | held_test | 0.0000 | 0.0000 | 0.0000 | 0.1270 | 0.4414 | 0.3144 | 0.2528 | 0.8933 | 0.0000 | 2.2865 |
| Qwen3-1.7B | 0 | held_test | 0.0000 | 0.0000 | 0.0000 | 0.0766 | 0.2846 | 0.2081 | 0.1573 | 0.5618 | 0.0000 | 2.4719 |
| Qwen3-1.7B | 4 | held_test | 0.0000 | 0.0000 | 0.0000 | 0.1240 | 0.4414 | 0.3174 | 0.2640 | 0.8933 | 0.0000 | 2.2809 |
| Qwen3-1.7B | 16 | held_test | 0.0000 | 0.0000 | 0.0000 | 0.1240 | 0.4414 | 0.3174 | 0.2640 | 0.8933 | 0.0000 | 2.2809 |
| Qwen3-4B | 0 | held_test | 0.0000 | 0.0000 | 0.0000 | 0.1062 | 0.4414 | 0.3352 | 0.2135 | 0.8933 | 0.0000 | 2.2809 |
| Qwen3-4B | 4 | held_test | 0.0000 | 0.0000 | 0.0000 | 0.0985 | 0.4492 | 0.3507 | 0.1966 | 0.8933 | 0.0056 | 2.3427 |
| Qwen3-4B | 16 | held_test | 0.0000 | 0.0000 | 0.0000 | 0.1059 | 0.4594 | 0.3534 | 0.2079 | 0.9045 | 0.0056 | 2.4831 |
| Qwen3-0.6B | 0 | held_control | 0.0000 | 0.0000 | 0.0000 | 0.0291 | 0.1644 | 0.1352 | 0.0467 | 0.2804 | 0.0093 | 2.4393 |
| Qwen3-0.6B | 4 | held_control | 0.0000 | 0.0093 | 0.0093 | 0.1031 | 0.3975 | 0.2944 | 0.1869 | 0.7290 | 0.0093 | 2.0280 |
| Qwen3-0.6B | 16 | held_control | 0.0000 | 0.0187 | 0.0187 | 0.1364 | 0.4022 | 0.2657 | 0.2523 | 0.7290 | 0.0187 | 2.0467 |
| Qwen3-1.7B | 0 | held_control | 0.0000 | 0.0000 | 0.0000 | 0.0489 | 0.1620 | 0.1131 | 0.0841 | 0.2897 | 0.0000 | 2.4486 |
| Qwen3-1.7B | 4 | held_control | 0.0000 | 0.0000 | 0.0000 | 0.1112 | 0.3944 | 0.2832 | 0.2056 | 0.7290 | 0.0000 | 2.0841 |
| Qwen3-1.7B | 16 | held_control | 0.0000 | 0.0000 | 0.0000 | 0.1112 | 0.3944 | 0.2832 | 0.2056 | 0.7290 | 0.0000 | 2.1028 |
| Qwen3-4B | 0 | held_control | 0.0000 | 0.0000 | 0.0000 | 0.1062 | 0.3944 | 0.2882 | 0.2056 | 0.7290 | 0.0000 | 2.1402 |
| Qwen3-4B | 4 | held_control | 0.0000 | 0.0000 | 0.0000 | 0.0988 | 0.3907 | 0.2919 | 0.1869 | 0.7196 | 0.0000 | 2.1682 |
| Qwen3-4B | 16 | held_control | 0.0000 | 0.0000 | 0.0000 | 0.1008 | 0.4206 | 0.3199 | 0.1776 | 0.7383 | 0.0280 | 2.3458 |

## Pretrained Interpretation

- The pretrained models are not failing on syntax. In the original scoring pipeline, `valid_rate` and `parseable_rate` were already uniformly `1.0`.
- The main pretrained failure is semantic grounding: tool identity is often recoverable, but magnitudes and multi-step composition are not.
- The clearest evidence is the gap between `first_tool_acc` and `first_call_acc`, plus the gap between strict and ignore-magnitude metrics.
- `Qwen3-1.7B` mostly saturates by `4` shots. `Qwen3-4B` benefits more from `16` shots than from `4` shots. `Qwen3-0.6B` gets the biggest raw prompting gain from `0 -> 4` shots.

## SFT Summary

This section adds the new SFT result into the same analysis framework. Only one SFT model is available so far: `outputs/qwen06b_sft_iid_18k`, trained on `iid_train` and evaluated on the same four `final18k` splits.

Average SFT metrics over the four splits: `exact=0.0493`, `exact_ignore_mag=0.3877`, `step_f1=0.2503`, `step_f1_ignore_mag=0.7019`, `first_call=0.3462`, `first_tool=0.8764`, `length_acc=0.4804`.

## SFT Split-Level Results

| split | total | exact | exact_ignore_mag | step_f1 | step_f1_ignore_mag | first_call | first_tool | length_acc | tool_edit_dist |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| iid_test | 1500 | 0.1240 | 0.5333 | 0.2815 | 0.7719 | 0.3547 | 0.9060 | 0.6047 | 0.7827 |
| len_test | 1100 | 0.0282 | 0.3636 | 0.2430 | 0.6889 | 0.3427 | 0.8718 | 0.4609 | 1.0673 |
| held_test | 601 | 0.0067 | 0.2729 | 0.1997 | 0.6407 | 0.2845 | 0.8619 | 0.3710 | 1.2696 |
| held_control | 365 | 0.0384 | 0.3808 | 0.2769 | 0.7062 | 0.4027 | 0.8658 | 0.4849 | 1.0164 |

## SFT vs `Qwen3-0.6B` 16-Shot Pretrained

This is the cleanest same-model comparison: the new IID SFT checkpoint against the strongest prompted pretrained version of the same model family that we ran today.

| split | d_exact | d_exact_ignore_mag | d_step_f1 | d_step_f1_ignore_mag | d_first_call | d_first_tool | d_length_acc | d_tool_edit_dist |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| iid_test | 0.0565 | 0.3583 | 0.1110 | 0.2746 | 0.0922 | 0.1085 | 0.3597 | -0.9098 |
| len_test | 0.0282 | 0.3571 | 0.1074 | 0.2582 | 0.0861 | 0.0462 | 0.4543 | -1.0676 |
| held_test | 0.0067 | 0.2729 | 0.0727 | 0.1994 | 0.0317 | -0.0314 | 0.3710 | -1.0170 |
| held_control | 0.0384 | 0.3621 | 0.1405 | 0.3040 | 0.1504 | 0.1368 | 0.4662 | -1.0303 |

## SFT vs Best Pretrained Configuration On Each Split

| split | metric | best_pretrained | SFT | delta |
|---|---|---|---:|---:|
| iid_test | exact_match | Qwen3-4B @ 16 (0.0725) | 0.1240 | 0.0515 |
| iid_test | exact_match_ignore_magnitude | Qwen3-4B @ 16 (0.2025) | 0.5333 | 0.3308 |
| iid_test | step_f1 | Qwen3-0.6B @ 16 (0.1705) | 0.2815 | 0.1110 |
| iid_test | step_f1_ignore_magnitude | Qwen3-4B @ 16 (0.5428) | 0.7719 | 0.2291 |
| iid_test | first_call_acc | Qwen3-0.6B @ 16 (0.2625) | 0.3547 | 0.0922 |
| iid_test | first_tool_acc | Qwen3-4B @ 16 (0.8375) | 0.9060 | 0.0685 |
| len_test | exact_match | Qwen3-0.6B @ 0 (0.0000) | 0.0282 | 0.0282 |
| len_test | exact_match_ignore_magnitude | Qwen3-0.6B @ 16 (0.0066) | 0.3636 | 0.3571 |
| len_test | step_f1 | Qwen3-0.6B @ 16 (0.1355) | 0.2430 | 0.1074 |
| len_test | step_f1_ignore_magnitude | Qwen3-4B @ 16 (0.4477) | 0.6889 | 0.2412 |
| len_test | first_call_acc | Qwen3-0.6B @ 16 (0.2566) | 0.3427 | 0.0861 |
| len_test | first_tool_acc | Qwen3-4B @ 16 (0.8355) | 0.8718 | 0.0363 |
| held_test | exact_match | Qwen3-0.6B @ 0 (0.0000) | 0.0067 | 0.0067 |
| held_test | exact_match_ignore_magnitude | Qwen3-0.6B @ 0 (0.0056) | 0.2729 | 0.2673 |
| held_test | step_f1 | Qwen3-0.6B @ 16 (0.1270) | 0.1997 | 0.0727 |
| held_test | step_f1_ignore_magnitude | Qwen3-4B @ 16 (0.4594) | 0.6407 | 0.1814 |
| held_test | first_call_acc | Qwen3-1.7B @ 4 (0.2640) | 0.2845 | 0.0205 |
| held_test | first_tool_acc | Qwen3-4B @ 16 (0.9045) | 0.8619 | -0.0426 |
| held_control | exact_match | Qwen3-0.6B @ 0 (0.0000) | 0.0384 | 0.0384 |
| held_control | exact_match_ignore_magnitude | Qwen3-0.6B @ 16 (0.0187) | 0.3808 | 0.3621 |
| held_control | step_f1 | Qwen3-0.6B @ 16 (0.1364) | 0.2769 | 0.1405 |
| held_control | step_f1_ignore_magnitude | Qwen3-4B @ 16 (0.4206) | 0.7062 | 0.2855 |
| held_control | first_call_acc | Qwen3-0.6B @ 16 (0.2523) | 0.4027 | 0.1504 |
| held_control | first_tool_acc | Qwen3-4B @ 16 (0.7383) | 0.8658 | 0.1274 |

## SFT Interpretation

### 1. SFT is a real improvement, not just a cosmetic one.

- On `iid_test`, SFT reaches `0.1240` strict exact match. The best pretrained result on the same split was `0.0725` from `Qwen3-4B @ 16`. That is a gain of `+0.0515` even against the strongest prompted pretrained baseline.
- On the same split, `exact_match_ignore_magnitude` jumps to `0.5333`, which is `+0.3308` above the best pretrained result and `+0.3583` above `Qwen3-0.6B @ 16`.
- This is strong evidence that SFT improves both magnitude grounding and multi-step structure, not just first-step tool selection.

### 2. The biggest SFT gains are on tool-sequence structure and magnitude grounding.

- On every split, the gain in ignore-magnitude exact match is much larger than the gain in strict exact match. That means SFT first stabilizes the correct action skeleton.
- But strict metrics also rise, especially on IID. So SFT is not only learning tool identities; it is partially learning the numeric argument mapping too.
- Example deltas versus `Qwen3-0.6B @ 16`: on `iid_test`, SFT adds `+0.1110` step F1 and `+0.2746` ignore-magnitude step F1. On `len_test`, the gains are `+0.1074` and `+0.2582` respectively.

### 3. SFT improves the first action, but the first-tool metric was already relatively easy.

- `first_tool_acc` was already high in the pretrained regime once few-shot prompting was used. So this is not the most diagnostic place to look for gains.
- Even so, SFT still improves `first_tool_acc` on `iid_test`, `len_test`, and `held_control`, and only loses on `held_test` relative to the single best pretrained first-tool baseline (`0.8619` vs `0.9045`).
- The more important result is `first_call_acc`: SFT improves it on all four splits, including `held_test` (`+0.0205` over the best pretrained configuration and `+0.0317` over `Qwen3-0.6B @ 16`).

### 4. IID-trained SFT still does not solve OOD generalization.

- `len_test` and `held_test` remain weak on strict exact match: `0.0282` and `0.0067` respectively.
- That is clearly better than the pretrained baselines, but it is still far from a solved system.
- The OOD results show that IID SFT helps the model internalize the DSL mapping much better, but not enough to fully generalize to longer or held-out compositional structures.

### 5. The length metric improves substantially under SFT.

- `length_acc` rises to `0.6047` on `iid_test`, `0.4609` on `len_test`, `0.3710` on `held_test`, and `0.4849` on `held_control`.
- Compared with `Qwen3-0.6B @ 16`, that is a gain of `+0.3597`, `+0.4543`, `+0.3710`, and `+0.4662` respectively.
- This matters because many pretrained failures looked like partial sequence collapse; SFT is substantially improving program-length control.

### 6. Bottom-line comparison

- Pretrained prompting is enough to get the model to emit valid DSL and often identify the right first tool.
- IID SFT changes the regime: the model is now meaningfully better at full program structure, program length, and magnitude recovery.
- But IID SFT alone still does not close the OOD gap. If the final paper wants to argue compositional failure, `held_test` remains the strongest evidence.

## Position-Wise Falloff: 1st vs 2nd vs 3rd Tool Call

This section measures how performance decays across the first three positions in the predicted program.
For each position `k`, metrics are computed over examples whose gold program has at least `k` calls.

Extra position-wise metrics:
- `pred_present_rate@k`: how often the model emitted a `k`-th action when the gold program has one.
- `call_acc@k`: exact match of the full `k`-th action, including magnitude.
- `tool_acc@k`: match of the `k`-th tool only, ignoring magnitude.
- `call_acc_given_present@k`: exact `k`-th action accuracy restricted to cases where the model emitted a `k`-th action.
- `tool_acc_given_present@k`: tool-only `k`-th action accuracy restricted to cases where the model emitted a `k`-th action.

### Why this matters

The sequence-level metrics already suggested that pretrained models often emit only one useful action and then collapse. Position-wise metrics let us separate two different failure modes:

1. The model stops too early and never emits the later call.
2. The model emits a later call, but it is the wrong tool or wrong magnitude.

### Best Pretrained Position-Wise Results

These are the strongest pretrained averages over the four splits for later-step prediction.

| position | best pretrained config | pred_present_rate | call_acc | tool_acc | call_acc_given_present | tool_acc_given_present |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `Qwen3-4B @ 16` for tool / `Qwen3-0.6B @ 16` for call | `1.0000` | `0.2561` | `0.8290` | `0.2561` | `0.8290` |
| 2 | `Qwen3-4B @ 16` | `0.0532` | `0.0101` | `0.0342` | `0.2027` | `0.6726` |
| 3 | `Qwen3-4B @ 16` | `0.0706` | `0.0160` | `0.0546` | `0.2143` | `0.7857` |

The key point is that pretrained later-step failure is dominated by early stopping.
The best pretrained setup still emits a second action only about `5.3%` of the time and a third action only about `7.1%` of the time, even when the gold sequence requires those positions.
When it does emit those later steps, the tool identity is not hopeless: `tool_acc_given_present` is `0.6726` at position 2 and `0.7857` at position 3.
So the main pretrained collapse is not “later tools are always wrong”; it is “later tools are usually never emitted.”

### SFT Position-Wise Results

Average over the four evaluated splits for `outputs/qwen06b_sft_iid_18k`:

| position | pred_present_rate | call_acc | tool_acc | call_acc_given_present | tool_acc_given_present |
|---:|---:|---:|---:|---:|---:|
| 1 | `1.0000` | `0.3462` | `0.8764` | `0.3462` | `0.8764` |
| 2 | `0.8287` | `0.2061` | `0.6348` | `0.2487` | `0.7658` |
| 3 | `0.5035` | `0.1212` | `0.3409` | `0.2443` | `0.6816` |

This is still a falloff, but it is a much healthier one.
Relative to position 1, SFT keeps:
- about `59.5%` of first-step call accuracy at position 2 and `35.0%` at position 3
- about `72.4%` of first-step tool accuracy at position 2 and `38.9%` at position 3
- about `82.9%` of required second steps and `50.4%` of required third steps

That is a completely different regime from the pretrained runs.

### SFT vs Best Pretrained Later-Step Comparison

Using the strongest pretrained position-wise baseline, `Qwen3-4B @ 16`, the SFT model gains:

| position | d_pred_present_rate | d_call_acc | d_tool_acc | d_call_acc_given_present | d_tool_acc_given_present |
|---:|---:|---:|---:|---:|---:|
| 1 | `0.0000` | `0.1448` | `0.0474` | `0.1448` | `0.0474` |
| 2 | `0.7755` | `0.1961` | `0.6006` | `0.0460` | `0.0932` |
| 3 | `0.4329` | `0.1052` | `0.2863` | `0.0300` | `-0.1041` |

Interpretation:
- The biggest gain is not conditional later-step accuracy. The biggest gain is that SFT actually keeps generating later steps.
- At position 2, SFT is about `+77.6` points higher in `pred_present_rate` than the best pretrained baseline.
- At position 3, SFT is about `+43.3` points higher in `pred_present_rate`.
- Conditional later-step tool accuracy is also somewhat better at position 2, but the main story is continuation rather than perfect late-step precision.
- The slight negative delta in `tool_acc_given_present` at position 3 should not be over-read: the pretrained model almost never reaches position 3, so that conditional number is based on a tiny, highly selected subset.

### Split-Level SFT Falloff

The SFT falloff pattern is consistent across splits:
- `iid_test`: tool accuracy goes `0.9060 -> 0.6373 -> 0.3383`, call accuracy goes `0.3547 -> 0.2073 -> 0.1173`
- `len_test`: tool accuracy goes `0.8718 -> 0.6373 -> 0.3383`, call accuracy goes `0.3427 -> 0.2073 -> 0.1173`
- `held_test`: tool accuracy goes `0.8619 -> 0.5990 -> 0.3272`, call accuracy goes `0.2845 -> 0.1963 -> 0.1002`
- `held_control`: tool accuracy goes `0.8658 -> 0.6658 -> 0.3597`, call accuracy goes `0.4027 -> 0.2137 -> 0.1502`

So even after SFT, the second and especially third action are harder than the first. But the degradation is gradual, not catastrophic.

### Bottom Line On Later-Step Decay

- In the pretrained regime, the second and third actions mostly fail because the model does not emit them at all.
- In the SFT regime, later actions still get worse, but mostly in the expected way: accuracy declines with sequence depth while continuation stays reasonably high.
- This strengthens the overall project story. Pretrained prompting is enough to identify the task format and often the first tool, but not enough to sustain a structured multi-step program. SFT substantially improves that continuation behavior.

## Combined Takeaways

1. The pretrained models from today are not failing because they cannot produce the output language. They fail because they do not robustly map language to the right structured program.
2. Magnitude grounding is the largest single pretrained weakness. The new ignore-magnitude metrics make that unambiguous.
3. Few-shot prompting helps a lot on tool identity, especially from `0 -> 4` shots, but it does not solve exact program generation.
4. IID SFT produces a real gain over the pretrained baselines, including over larger prompted models on IID.
5. IID SFT still leaves substantial OOD failure, so the final project can cleanly separate “learned the DSL mapping” from “generalized compositionally.”

## Files

- Pretrained rescored metrics JSONL: `outputs/pretrained_full_rescored_extra_metrics.jsonl`
- Pretrained rescored metrics CSV: `outputs/pretrained_full_rescored_extra_metrics.csv`
- SFT rescored metrics JSONL: `outputs/final18k_rescored_extra_metrics.jsonl`
- SFT rescored metrics CSV: `outputs/final18k_rescored_extra_metrics.csv`
- This analysis document: `outputs/NON_SFT_TODAY_EXTRA_EVAL_ANALYSIS.md`
- Pretrained raw prediction source: `finalproj_outputs/outputs/final_raw/pre_qwen*.jsonl`
- SFT checkpoint metadata: `finalproj_outputs/outputs/qwen06b_sft_iid_18k/run_metadata.json`
- SFT raw prediction source: `finalproj_outputs/outputs/final_raw/qwen06b_sft_iid_18k_outputs_qwen06b_sft_iid_18k_data_final18k_*.jsonl`