import argparse
import inspect
from dataclasses import dataclass
from typing import Dict, List

import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments

from .utils import load_model_and_tokenizer, read_jsonl

PROMPT_TMPL = "Instruction: {instr}\nAction sequence:"

class JsonlProgramDataset(Dataset):
    def __init__(self, rows: List[Dict], tokenizer, max_length: int = 128, mask_output: bool = True):
        self.rows = rows
        self.tok = tokenizer
        self.max_length = max_length
        self.mask_output = mask_output

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        prompt = PROMPT_TMPL.format(instr=r["instruction"])
        target = r["program"].strip()

        # tokenize prompt and full input
        prompt_ids = self.tok(prompt, add_special_tokens=False).input_ids
        full = prompt + " " + target
        enc = self.tok(full, truncation=True, max_length=self.max_length, padding=False, return_tensors=None)
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        labels = input_ids.copy()
        if self.mask_output:
            # mask prompt tokens
            prompt_len = min(len(prompt_ids), len(labels))
            for i in range(prompt_len):
                labels[i] = -100

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

@dataclass
class Collator:
    pad_token_id: int
    def __call__(self, batch):
        # pad to max length in batch
        max_len = max(x["input_ids"].shape[0] for x in batch)
        input_ids, attn, labels = [], [], []
        for x in batch:
            L = x["input_ids"].shape[0]
            pad = max_len - L
            input_ids.append(torch.cat([x["input_ids"], torch.full((pad,), self.pad_token_id, dtype=torch.long)]))
            attn.append(torch.cat([x["attention_mask"], torch.zeros((pad,), dtype=torch.long)]))
            labels.append(torch.cat([x["labels"], torch.full((pad,), -100, dtype=torch.long)]))
        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attn),
            "labels": torch.stack(labels),
        }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="gpt2")
    ap.add_argument("--train", type=str, required=True)
    ap.add_argument("--dev", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--mask_output", type=int, default=1)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--bsz", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    args = ap.parse_args()

    if args.fp16 and args.bf16:
        raise ValueError("Choose only one of --fp16 or --bf16.")

    model, tok = load_model_and_tokenizer(args.model)
    param_dtype = next(model.parameters()).dtype

    use_fp16 = bool(args.fp16)
    use_bf16 = bool(args.bf16)

    # Some Qwen checkpoints load in bf16 by default; fp16 GradScaler can fail on that.
    if use_fp16 and not use_bf16 and param_dtype == torch.bfloat16:
        print("[warn] Model parameters loaded as bfloat16; switching training AMP mode from fp16 -> bf16.")
        use_fp16 = False
        use_bf16 = True

    train_rows = read_jsonl(args.train)
    dev_rows = read_jsonl(args.dev)

    train_ds = JsonlProgramDataset(train_rows, tok, max_length=args.max_length, mask_output=bool(args.mask_output))
    dev_ds = JsonlProgramDataset(dev_rows, tok, max_length=args.max_length, mask_output=bool(args.mask_output))

    collator = Collator(pad_token_id=tok.pad_token_id)

    targs_kwargs = dict(
        output_dir=args.out,
        per_device_train_batch_size=args.bsz,
        per_device_eval_batch_size=args.bsz,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=50,
        save_steps=500,
        eval_steps=500,
        save_total_limit=2,
        report_to=[],
        fp16=use_fp16,
        bf16=use_bf16,
        remove_unused_columns=False,
    )

    # HF versions differ: some expect `evaluation_strategy`, newer ones use `eval_strategy`.
    targs_params = inspect.signature(TrainingArguments.__init__).parameters
    if "evaluation_strategy" in targs_params:
        targs_kwargs["evaluation_strategy"] = "steps"
    elif "eval_strategy" in targs_params:
        targs_kwargs["eval_strategy"] = "steps"

    targs = TrainingArguments(**targs_kwargs)

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print(f"Saved to {args.out}")

if __name__ == "__main__":
    main()
