# src/eval.py
import argparse
from typing import Dict, List, Tuple
from collections import Counter

import torch

from .utils import load_model_and_tokenizer, read_jsonl, make_prefix_allowed_tokens_fn
from .actions import is_valid, parse, parse_loose, extract_program_prefix
from .sim import execute, trajectory_score

PROMPT_TMPL = "Instruction: {instr}\nAction sequence:"

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

@torch.no_grad()
def generate_program(model, tok, instruction: str, constrained: bool, max_new_tokens: int, temperature: float):
    prompt = PROMPT_TMPL.format(instr=instruction)
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
        prog = completion.strip().splitlines()[0].strip()
    return prog

def eval_file(model_path: str, test_path: str, constrained: bool, max_new_tokens: int, temperature: float):
    model, tok = load_model_and_tokenizer(model_path)
    model.eval()

    rows = read_jsonl(test_path)

    metrics = Counter()
    precs, recs, f1s = [], [], []
    edit_tools = []
    length_ok = 0
    traj_scores = []

    for r in rows:
        gold_prog = r["program"].strip()
        gold_actions = parse(gold_prog)

        pred_prog = generate_program(model, tok, r["instruction"], constrained, max_new_tokens, temperature)

        metrics["total"] += 1
        strict_valid = is_valid(pred_prog)
        if strict_valid:
            metrics["valid"] += 1
        
        pred_actions = parse_loose(pred_prog)  # best-effort parse

        if pred_actions is not None and gold_actions is not None:
            # exact match (canonical program string match)
            if pred_prog.strip() == gold_prog:
                metrics["exact"] += 1

            p, rr, f1 = step_f1(pred_actions, gold_actions)
            precs.append(p); recs.append(rr); f1s.append(f1)

            pred_tools = [t for (t,_) in pred_actions]
            gold_tools = [t for (t,_) in gold_actions]
            edit_tools.append(levenshtein(pred_tools, gold_tools))

            if len(pred_actions) == len(gold_actions):
                length_ok += 1

            traj = execute(pred_actions)
            traj_ref = execute(gold_actions)
            traj_scores.append(trajectory_score(traj, traj_ref))
        else:
            # invalid program or parse failure
            pass

    total = metrics["total"]
    valid = metrics["valid"]
    exact = metrics["exact"]

    out = {
        "total": total,
        "valid_rate": valid / max(total,1),
        "exact_match": exact / max(total,1),
        "step_precision": sum(precs)/max(len(precs),1),
        "step_recall": sum(recs)/max(len(recs),1),
        "step_f1": sum(f1s)/max(len(f1s),1),
        "tool_edit_dist": sum(edit_tools)/max(len(edit_tools),1),
        "length_acc": length_ok / max(total,1),
        "mean_traj_score": sum(traj_scores)/max(len(traj_scores),1),
    }
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--test", type=str, required=True)
    ap.add_argument("--constrained", type=int, default=0)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.0)  # 0 => greedy
    args = ap.parse_args()

    res = eval_file(args.model, args.test, bool(args.constrained), args.max_new_tokens, args.temperature)
    print(res)

if __name__ == "__main__":
    main()