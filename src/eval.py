# src/eval.py
import argparse
import random
from typing import Dict, List, Tuple
from collections import Counter

import torch

from .utils import load_model_and_tokenizer, read_jsonl, make_prefix_allowed_tokens_fn
from .actions import is_valid, parse, parse_loose, extract_program_prefix, first_invalid_reason, serialize

PROMPT_TMPL = "Instruction: {instr}\nAction sequence:"
DEMO_TMPL = "Instruction: {instr}\nAction sequence: {prog}"
TASK_SPEC = (
    "You are converting instructions into an action sequence program.\n"
    "Output only the program, with no extra text.\n"
    "Program format: concatenate bracketed calls with no separators.\n"
    "Each call format: [tool:value]\n"
    "Allowed tools: forward, backward, left, right, bark\n"
    "Allowed values for forward/backward: 10, 30, 60, 100\n"
    "Allowed values for left/right: 15, 45, 90, 180\n"
    "Allowed value for bark: 0\n"
    "Example format only: [left:45][forward:60][bark:0]"
)

def levenshtein(a: List[str], b: List[str]) -> int:
    # simple DP for small sequences
    n, m = len(a), len(b)
    dp = list(range(m+1))
    for i in range(1, n+1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m+1):
            cur = dp[j]
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[j] = min(dp[j] + 1, dp[j-1] + 1, prev + cost)
            prev = cur
    return dp[m]

def step_f1(pred: List[Tuple[str,int]], gold: List[Tuple[str,int]]) -> Tuple[float,float,float]:
    # treat each step positionally; compute precision/recall on exact (tool,val) matches at each position
    # alternative: multiset; but positional is more meaningful here
    L = max(len(pred), len(gold))
    if L == 0:
        return 1.0, 1.0, 1.0
    tp = 0
    for i in range(min(len(pred), len(gold))):
        if pred[i] == gold[i]:
            tp += 1
    prec = tp / max(len(pred), 1)
    rec = tp / max(len(gold), 1)
    f1 = 0.0 if (prec+rec)==0 else (2*prec*rec)/(prec+rec)
    return prec, rec, f1

def tool_step_f1(pred: List[Tuple[str,int]], gold: List[Tuple[str,int]]) -> Tuple[float,float,float]:
    pred_tools = [t for (t, _) in pred]
    gold_tools = [t for (t, _) in gold]
    L = max(len(pred_tools), len(gold_tools))
    if L == 0:
        return 1.0, 1.0, 1.0
    tp = 0
    for i in range(min(len(pred_tools), len(gold_tools))):
        if pred_tools[i] == gold_tools[i]:
            tp += 1
    prec = tp / max(len(pred_tools), 1)
    rec = tp / max(len(gold_tools), 1)
    f1 = 0.0 if (prec + rec) == 0 else (2 * prec * rec) / (prec + rec)
    return prec, rec, f1

def build_fewshot_prefix(rows: List[Dict], num_shots: int, seed: int) -> str:
    if num_shots <= 0 or len(rows) == 0:
        return ""
    rng = random.Random(seed)
    idxs = list(range(len(rows)))
    rng.shuffle(idxs)
    demos = []
    for i in idxs[:min(num_shots, len(rows))]:
        r = rows[i]
        demos.append(DEMO_TMPL.format(instr=str(r["instruction"]).strip(), prog=str(r["program"]).strip()))
    return "\n\n".join(demos) + "\n\n"

@torch.no_grad()
def generate_program(
    model,
    tok,
    instruction: str,
    constrained: bool,
    max_new_tokens: int,
    temperature: float,
    fewshot_prefix: str = "",
    include_task_spec: bool = True,
):
    prompt_parts = []
    if include_task_spec:
        prompt_parts.append(TASK_SPEC)
    if fewshot_prefix:
        prompt_parts.append(fewshot_prefix.rstrip())
    prompt_parts.append(PROMPT_TMPL.format(instr=instruction))
    prompt = "\n\n".join(prompt_parts)
    inputs = tok(prompt, return_tensors="pt")
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        pad_token_id=tok.eos_token_id,
    )
    if temperature <= 0:
        gen_kwargs.update(dict(do_sample=False))
    else:
        gen_kwargs.update(dict(do_sample=True, temperature=temperature, top_p=0.95))

    if constrained:
        gen_kwargs["prefix_allowed_tokens_fn"] = make_prefix_allowed_tokens_fn(tok)

    out = model.generate(**inputs, **gen_kwargs)
    prompt_len = inputs["input_ids"].shape[1]
    completion = tok.decode(out[0][prompt_len:], skip_special_tokens=True)
    prog = extract_program_prefix(completion)
    if prog is None:
        # Fall back to a tolerant parser (handles leading model chatter before calls).
        acts = parse_loose(completion)
        if acts is not None:
            prog = serialize(acts)
    if prog is None:
        prog = completion.strip().splitlines()[0].strip()
    return prog

def eval_file(
    model_path: str,
    test_path: str,
    constrained: bool,
    max_new_tokens: int,
    temperature: float,
    fewshot_path: str = "",
    num_shots: int = 0,
    fewshot_seed: int = 0,
    include_task_spec: bool = True,
):
    model, tok = load_model_and_tokenizer(model_path)
    model.eval()

    rows = read_jsonl(test_path)
    fewshot_rows = read_jsonl(fewshot_path) if fewshot_path else []
    fewshot_prefix = build_fewshot_prefix(fewshot_rows, num_shots=num_shots, seed=fewshot_seed)

    metrics = Counter()
    precs, recs, f1s = [], [], []
    tprecs, trecs, tf1s = [], [], []
    edit_tools = []
    length_ok = 0
    invalid_reasons = Counter()

    for r in rows:
        gold_prog = r["program"].strip()
        gold_actions = parse(gold_prog)

        pred_prog = generate_program(
            model,
            tok,
            r["instruction"],
            constrained,
            max_new_tokens,
            temperature,
            fewshot_prefix=fewshot_prefix,
            include_task_spec=include_task_spec,
        )

        metrics["total"] += 1
        strict_valid = is_valid(pred_prog)
        if strict_valid:
            metrics["valid"] += 1
        else:
            invalid_reasons[first_invalid_reason(pred_prog)] += 1
        
        pred_actions = parse_loose(pred_prog)  # best-effort parse
        if pred_actions is not None:
            metrics["parseable"] += 1

        if pred_actions is not None and gold_actions is not None:
            # exact match (canonical program string match)
            if pred_prog.strip() == gold_prog:
                metrics["exact"] += 1

            p, rr, f1 = step_f1(pred_actions, gold_actions)
            precs.append(p); recs.append(rr); f1s.append(f1)

            pred_tools = [t for (t,_) in pred_actions]
            gold_tools = [t for (t,_) in gold_actions]
            tp, tr, tf1 = tool_step_f1(pred_actions, gold_actions)
            tprecs.append(tp); trecs.append(tr); tf1s.append(tf1)
            if pred_tools == gold_tools:
                metrics["tool_exact"] += 1
            edit_tools.append(levenshtein(pred_tools, gold_tools))

            if len(pred_actions) == len(gold_actions):
                length_ok += 1
        else:
            # invalid program or parse failure
            pass

    total = metrics["total"]
    valid = metrics["valid"]
    exact = metrics["exact"]

    out = {
        "total": total,
        "valid_rate": valid / max(total,1),
        "parseable_rate": metrics["parseable"] / max(total,1),
        "exact_match": exact / max(total,1),
        "tool_exact_match": metrics["tool_exact"] / max(total,1),
        "step_precision": sum(precs)/max(len(precs),1),
        "step_recall": sum(recs)/max(len(recs),1),
        "step_f1": sum(f1s)/max(len(f1s),1),
        "tool_step_precision": sum(tprecs)/max(len(tprecs),1),
        "tool_step_recall": sum(trecs)/max(len(trecs),1),
        "tool_step_f1": sum(tf1s)/max(len(tf1s),1),
        "tool_edit_dist": sum(edit_tools)/max(len(edit_tools),1),
        "length_acc": length_ok / max(total,1),
        "num_shots": int(num_shots),
        "include_task_spec": int(bool(include_task_spec)),
        "invalid_reasons": dict(invalid_reasons),
    }
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--test", type=str, required=True)
    ap.add_argument("--constrained", type=int, default=0)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.0)  # 0 => greedy
    ap.add_argument("--fewshot_path", type=str, default="")
    ap.add_argument("--num_shots", type=int, default=0)
    ap.add_argument("--fewshot_seed", type=int, default=0)
    ap.add_argument("--include_task_spec", type=int, default=1)
    args = ap.parse_args()

    res = eval_file(
        args.model,
        args.test,
        bool(args.constrained),
        args.max_new_tokens,
        args.temperature,
        fewshot_path=args.fewshot_path,
        num_shots=args.num_shots,
        fewshot_seed=args.fewshot_seed,
        include_task_spec=bool(args.include_task_spec),
    )
    print(res)

if __name__ == "__main__":
    main()
