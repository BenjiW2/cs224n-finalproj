import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from actions import parse, is_valid
from sim import execute, trajectory_score

PROMPTTEMP = "Instruction: {instruct}\nAction sequence:"

def generate_candidates(model, tokenizer, instr, M=8, max_new_tokens=64):
    prompt = PROMPTTEMP.format(instruct=instr)
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            num_return_sequences=M,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )
    # decode only the generated completion after the prompt
    prompt_len = inputs["input_ids"].shape[1]
    cands = []
    for seq in out:
        completion = tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
        # often includes extra whitespace/newlines; strip
        prog = completion.strip().splitlines()[0].strip()
        cands.append(prog)
    return cands

def self_train(
    base_ckpt="outputs/sft_base",
    unlabeled_jsonl="data/unlabeled_len234.jsonl",
    out_pseudo="data/pseudo_iter1.jsonl",
    M=8,
    tau=0.85,
):
    tokenizer = AutoTokenizer.from_pretrained(base_ckpt)
    model = AutoModelForCausalLM.from_pretrained(base_ckpt)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id
    model.eval()

    kept = 0
    total = 0

    with open(unlabeled_jsonl, "r", encoding="utf-8") as f_in, open(out_pseudo, "w", encoding="utf-8") as f_out:
        for line in f_in:
            ex = json.loads(line)
            instr = ex["instruction"]
            gt_prog = ex["program"]  # hidden from model, used only for scoring

            gt_actions = parse(gt_prog)
            if gt_actions is None:
                continue
            traj_ref = execute(gt_actions)

            cands = generate_candidates(model, tokenizer, instr, M=M)

            best = None
            best_u = -1.0
            for prog in cands:
                if not is_valid(prog):
                    continue
                actions = parse(prog)
                if actions is None:
                    continue
                traj = execute(actions)
                u = trajectory_score(traj, traj_ref)
                if u > best_u:
                    best_u = u
                    best = prog

            total += 1
            if best is not None and best_u >= tau:
                kept += 1
                f_out.write(json.dumps({
                    "instruction": instr,
                    "program": best,
                    "utility": float(best_u),
                }) + "\n")

    print(f"Kept {kept}/{total} = {kept/total:.3f}")

if __name__ == "__main__":
    self_train()