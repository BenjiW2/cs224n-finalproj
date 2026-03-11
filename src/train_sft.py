import argparse
import csv
import inspect
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments

from .actions import serialize
from .utils import load_model_and_tokenizer, read_jsonl

PROMPT_TMPL = "Instruction: {instr}\nAction sequence:"
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


def save_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def save_jsonl(path: str, rows: List[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_csv(path: str, rows: List[Dict]) -> None:
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([])
        return
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_run_metadata(args, train_rows: List[Dict], dev_rows: List[Dict]) -> Dict:
    effective_batch_size = args.bsz * args.grad_accum
    steps_per_epoch = math.ceil(len(train_rows) / effective_batch_size) if train_rows else 0
    total_optimizer_steps = math.ceil(steps_per_epoch * args.epochs)
    return {
        "model": args.model,
        "train_path": args.train,
        "dev_path": args.dev,
        "output_dir": args.out,
        "train_examples": len(train_rows),
        "dev_examples": len(dev_rows),
        "per_device_batch_size": args.bsz,
        "gradient_accumulation_steps": args.grad_accum,
        "effective_batch_size": effective_batch_size,
        "epochs": args.epochs,
        "estimated_steps_per_epoch": steps_per_epoch,
        "estimated_total_optimizer_steps": total_optimizer_steps,
        "max_length": args.max_length,
        "learning_rate": args.lr,
        "mask_output": bool(args.mask_output),
        "fp16": bool(args.fp16),
        "bf16": bool(args.bf16),
        "include_task_spec": bool(args.include_task_spec),
        "use_lora": bool(args.use_lora),
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "lora_target_modules": args.lora_target_modules,
        "gradient_checkpointing": bool(args.gradient_checkpointing),
        "eval_strategy": args.eval_strategy,
        "save_strategy": args.save_strategy,
        "save_total_limit": args.save_total_limit,
        "logging_steps": args.logging_steps,
        "eval_steps": args.eval_steps,
        "save_steps": args.save_steps,
        "length_weights": args.length_weights,
        "step_weights": args.step_weights,
    }


def build_loss_rows(log_history: List[Dict]) -> List[Dict]:
    keep = ("step", "epoch", "loss", "eval_loss", "learning_rate", "grad_norm")
    rows = []
    for event in log_history:
        row = {}
        for key in keep:
            if key in event:
                row[key] = event[key]
        if row:
            rows.append(row)
    return rows


def build_prompt(instr: str, include_task_spec: bool = False) -> str:
    prompt = PROMPT_TMPL.format(instr=instr)
    if include_task_spec:
        return TASK_SPEC + "\n\n" + prompt
    return prompt


def parse_weight_list(spec: str, name: str, expected_len: int = 4) -> List[float]:
    vals = [float(x.strip()) for x in spec.split(",") if x.strip()]
    if len(vals) != expected_len:
        raise ValueError(f"{name} must have exactly {expected_len} comma-separated floats.")
    return vals


def weighting_is_active(length_weights: List[float], step_weights: List[float]) -> bool:
    return any(abs(x - 1.0) > 1e-8 for x in length_weights + step_weights)

class JsonlProgramDataset(Dataset):
    def __init__(
        self,
        rows: List[Dict],
        tokenizer,
        max_length: int = 128,
        mask_output: bool = True,
        include_task_spec: bool = False,
        length_weights: Optional[List[float]] = None,
        step_weights: Optional[List[float]] = None,
    ):
        self.rows = rows
        self.tok = tokenizer
        self.max_length = max_length
        self.mask_output = mask_output
        self.include_task_spec = include_task_spec
        self.length_weights = length_weights or [1.0, 1.0, 1.0, 1.0]
        self.step_weights = step_weights or [1.0, 1.0, 1.0, 1.0]
        self.weighted_loss = weighting_is_active(self.length_weights, self.step_weights)

    def __len__(self):
        return len(self.rows)

    def _build_loss_weights(
        self,
        row: Dict,
        prompt: str,
        input_ids: List[int],
        labels: List[int],
    ) -> List[float]:
        weights = [0.0] * len(input_ids)
        if not self.weighted_loss:
            return weights

        length = int(row.get("length", 1))
        ex_weight = self.length_weights[min(max(length, 1), len(self.length_weights)) - 1]

        prompt_core_len = len(self.tok(prompt, add_special_tokens=False).input_ids)
        prefix_text = prompt + " "
        prefix_len = len(self.tok(prefix_text, add_special_tokens=False).input_ids)

        # Preserve previous training behavior: the separating space before the program
        # still contributes to loss and is weighted like step 1.
        leading_step_weight = ex_weight * self.step_weights[0]
        for pos in range(prompt_core_len, min(prefix_len, len(weights))):
            if labels[pos] != -100:
                weights[pos] = leading_step_weight

        actions = [(str(t), int(v)) for (t, v) in row.get("actions", [])]
        if not actions:
            for pos in range(prefix_len, len(weights)):
                if labels[pos] != -100:
                    weights[pos] = ex_weight
            return weights

        cursor = prefix_text
        last_end = min(prefix_len, len(weights))
        for step_idx, action in enumerate(actions):
            prev_len = len(self.tok(cursor, add_special_tokens=False).input_ids)
            cursor = cursor + serialize([action])
            next_len = len(self.tok(cursor, add_special_tokens=False).input_ids)
            step_weight = ex_weight * self.step_weights[min(step_idx, len(self.step_weights) - 1)]
            for pos in range(prev_len, min(next_len, len(weights))):
                if labels[pos] != -100:
                    weights[pos] = step_weight
            last_end = min(next_len, len(weights))

        # If truncation or tokenizer edge cases left any target tokens unassigned, give them
        # the final step's weight rather than silently dropping them.
        fallback_weight = ex_weight * self.step_weights[min(len(actions) - 1, len(self.step_weights) - 1)]
        for pos in range(last_end, len(weights)):
            if labels[pos] != -100 and weights[pos] == 0.0:
                weights[pos] = fallback_weight
        return weights

    def __getitem__(self, idx):
        r = self.rows[idx]
        prompt = build_prompt(str(r["instruction"]), include_task_spec=self.include_task_spec)
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

        item = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
        if self.weighted_loss:
            loss_weights = self._build_loss_weights(r, prompt, input_ids, labels)
            item["loss_weights"] = torch.tensor(loss_weights, dtype=torch.float32)
        return item

@dataclass
class Collator:
    pad_token_id: int
    def __call__(self, batch):
        # pad to max length in batch
        max_len = max(x["input_ids"].shape[0] for x in batch)
        input_ids, attn, labels = [], [], []
        loss_weights = []
        has_loss_weights = "loss_weights" in batch[0]
        for x in batch:
            L = x["input_ids"].shape[0]
            pad = max_len - L
            input_ids.append(torch.cat([x["input_ids"], torch.full((pad,), self.pad_token_id, dtype=torch.long)]))
            attn.append(torch.cat([x["attention_mask"], torch.zeros((pad,), dtype=torch.long)]))
            labels.append(torch.cat([x["labels"], torch.full((pad,), -100, dtype=torch.long)]))
            if has_loss_weights:
                loss_weights.append(torch.cat([x["loss_weights"], torch.zeros((pad,), dtype=torch.float32)]))
        out = {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attn),
            "labels": torch.stack(labels),
        }
        if has_loss_weights:
            out["loss_weights"] = torch.stack(loss_weights)
        return out


class WeightedLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss_weights = inputs.pop("loss_weights", None)
        outputs = model(**inputs)
        if loss_weights is None:
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
            return (loss, outputs) if return_outputs else loss

        logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
        labels = inputs["labels"]
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_weights = loss_weights[..., 1:].contiguous().to(shift_logits.device)

        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
        token_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        ).view_as(shift_labels)

        valid = shift_labels.ne(-100).float()
        weighted = token_loss * shift_weights * valid
        denom = (shift_weights * valid).sum().clamp(min=1e-8)
        loss = weighted.sum() / denom
        return (loss, outputs) if return_outputs else loss

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
    ap.add_argument("--logging_steps", type=int, default=10)
    ap.add_argument("--eval_steps", type=int, default=50)
    ap.add_argument("--save_steps", type=int, default=200)
    ap.add_argument("--eval_strategy", type=str, default="steps", choices=["steps", "epoch", "no"])
    ap.add_argument("--save_strategy", type=str, default="steps", choices=["steps", "epoch", "no"])
    ap.add_argument("--save_total_limit", type=int, default=2)
    ap.add_argument("--include_task_spec", type=int, default=0)
    ap.add_argument("--use_lora", action="store_true")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    ap.add_argument("--gradient_checkpointing", action="store_true")
    ap.add_argument(
        "--length_weights",
        type=str,
        default="1.0,1.0,1.0,1.0",
        help="Per-example weights for program lengths 1,2,3,4.",
    )
    ap.add_argument(
        "--step_weights",
        type=str,
        default="1.0,1.0,1.0,1.0",
        help="Per-step weights for action calls 1,2,3,4.",
    )
    args = ap.parse_args()

    if args.fp16 and args.bf16:
        raise ValueError("Choose only one of --fp16 or --bf16.")

    length_weights = parse_weight_list(args.length_weights, "--length_weights")
    step_weights = parse_weight_list(args.step_weights, "--step_weights")

    model, tok = load_model_and_tokenizer(args.model)
    param_dtype = next(model.parameters()).dtype

    use_fp16 = bool(args.fp16)
    use_bf16 = bool(args.bf16)

    # Some Qwen checkpoints load in bf16 by default; fp16 GradScaler can fail on that.
    if use_fp16 and not use_bf16 and param_dtype == torch.bfloat16:
        print("[warn] Model parameters loaded as bfloat16; switching training AMP mode from fp16 -> bf16.")
        use_fp16 = False
        use_bf16 = True

    if args.use_lora:
        from peft import LoraConfig, get_peft_model

        target_modules = [x.strip() for x in args.lora_target_modules.split(",") if x.strip()]
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    train_rows = read_jsonl(args.train)
    dev_rows = read_jsonl(args.dev)

    train_ds = JsonlProgramDataset(
        train_rows,
        tok,
        max_length=args.max_length,
        mask_output=bool(args.mask_output),
        include_task_spec=bool(args.include_task_spec),
        length_weights=length_weights,
        step_weights=step_weights,
    )
    dev_ds = JsonlProgramDataset(
        dev_rows,
        tok,
        max_length=args.max_length,
        mask_output=bool(args.mask_output),
        include_task_spec=bool(args.include_task_spec),
        length_weights=length_weights,
        step_weights=step_weights,
    )

    run_meta = build_run_metadata(args, train_rows, dev_rows)
    os.makedirs(args.out, exist_ok=True)
    print(json.dumps({"run_metadata": run_meta}, indent=2))
    save_json(os.path.join(args.out, "run_metadata.json"), run_meta)

    collator = Collator(pad_token_id=tok.pad_token_id)

    targs_kwargs = dict(
        output_dir=args.out,
        per_device_train_batch_size=args.bsz,
        per_device_eval_batch_size=args.bsz,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,
        report_to=[],
        fp16=use_fp16,
        bf16=use_bf16,
        remove_unused_columns=False,
        logging_first_step=True,
        gradient_checkpointing=bool(args.gradient_checkpointing),
    )

    # HF versions differ: some expect `evaluation_strategy`, newer ones use `eval_strategy`.
    targs_params = inspect.signature(TrainingArguments.__init__).parameters
    if "evaluation_strategy" in targs_params:
        targs_kwargs["evaluation_strategy"] = args.eval_strategy
    elif "eval_strategy" in targs_params:
        targs_kwargs["eval_strategy"] = args.eval_strategy

    if "save_strategy" in targs_params:
        targs_kwargs["save_strategy"] = args.save_strategy

    if args.eval_strategy == "steps":
        targs_kwargs["eval_steps"] = args.eval_steps
    if args.save_strategy == "steps":
        targs_kwargs["save_steps"] = args.save_steps

    targs = TrainingArguments(**targs_kwargs)

    trainer = WeightedLossTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=collator,
    )

    train_result = trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    trainer.state.save_to_json(os.path.join(args.out, "trainer_state.json"))

    log_history = trainer.state.log_history
    loss_rows = build_loss_rows(log_history)
    save_jsonl(os.path.join(args.out, "training_log.jsonl"), log_history)
    save_csv(os.path.join(args.out, "training_log.csv"), log_history)
    save_jsonl(os.path.join(args.out, "loss_curve.jsonl"), loss_rows)
    save_csv(os.path.join(args.out, "loss_curve.csv"), loss_rows)

    train_metrics = dict(train_result.metrics)
    save_json(os.path.join(args.out, "train_metrics.json"), train_metrics)
    print(json.dumps({"train_metrics": train_metrics}, indent=2))
    print(f"Saved training log to {os.path.join(args.out, 'training_log.csv')}")
    print(f"Saved loss curve to {os.path.join(args.out, 'loss_curve.csv')}")
    print(f"Saved to {args.out}")

if __name__ == "__main__":
    main()
