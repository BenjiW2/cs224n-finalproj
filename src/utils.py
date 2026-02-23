import json
from typing import Dict, List, Callable, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM

SPECIAL_TOKENS = [
    "[", "]", ":", "forward", "backward", "left", "right", "bark",
    "10", "30", "60", "100", "15", "45", "90", "180", "0"
]

def load_model_and_tokenizer(model_name_or_path: str):
    tok = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)

    # GPT-2 padding
    tok.pad_token = tok.eos_token
    model.config.pad_token_id = tok.eos_token_id

    # Only add truly missing DSL tokens, and add them as normal tokens
    # (not special tokens) so decoding does not strip them.
    missing = []
    for t in SPECIAL_TOKENS:
        tid = tok.convert_tokens_to_ids(t)
        if tid is None:
            missing.append(t)
            continue
        if tok.unk_token_id is not None and tid == tok.unk_token_id and t != tok.unk_token:
            missing.append(t)
    if missing:
        tok.add_tokens(missing, special_tokens=False)
        model.resize_token_embeddings(len(tok))
    return model, tok

def read_jsonl(path: str) -> List[Dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data

def write_jsonl(path: str, rows: List[Dict]):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# ---------------- Constrained decoding FSM ----------------
# We constrain generation to token sequence:
#   "[" tool ":" value "]" (repeat) then EOS
# tool ∈ move tools + bark
# value ∈ {10,30,60,100} for forward/backward
# value ∈ {15,45,90,180} for left/right
# value ∈ {0} for bark

def make_prefix_allowed_tokens_fn(tokenizer, prompt_len: int = 0) -> Callable:
    tid = {t: tokenizer.convert_tokens_to_ids(t) for t in SPECIAL_TOKENS}
    missing = [k for (k, v) in tid.items() if v is None]
    if missing:
        raise ValueError(f"Missing DSL token ids in tokenizer: {missing}")

    prompt_len = max(int(prompt_len), 0)
    eos = tokenizer.eos_token_id

    TOOL_IDS = [tid["forward"], tid["backward"], tid["left"], tid["right"], tid["bark"]]
    DIST_TOOL_IDS = [tid["forward"], tid["backward"]]
    TURN_TOOL_IDS = [tid["left"], tid["right"]]
    VAL_DIST_IDS = [tid["10"], tid["30"], tid["60"], tid["100"]]
    VAL_TURN_IDS = [tid["15"], tid["45"], tid["90"], tid["180"]]
    VAL_BARK_ID = tid["0"]

    LBR = tid["["]
    RBR = tid["]"]
    COL = tid[":"]

    # FSM states:
    # 0: expect "[" or EOS (EOS only allowed if already completed at least 1 call)
    # 1: expect TOOL
    # 2: expect ":"
    # 3: expect VALUE (depends on tool)
    # 4: expect "]"
    # then back to 0
    #
    # We'll track "tool_type" in state by encoding as:
    #   state=3 with move or bark; we store last tool id in a closure via parsing the prefix tokens.

    def allowed(batch_id: int, input_ids) -> List[int]:
        # `transformers` may pass either:
        # - a batched tensor/list (shape [batch, seq]) or
        # - a single sequence tensor/list (shape [seq]).
        # Normalize both cases to a flat Python token-id list.
        if hasattr(input_ids, "tolist"):
            raw = input_ids.tolist()
        else:
            raw = input_ids

        if isinstance(raw, list) and raw and isinstance(raw[0], list):
            seq = raw[batch_id]
        elif isinstance(raw, list):
            seq = raw
        else:
            seq = [int(raw)]

        # Constrain only on the generated suffix, never on prompt tokens.
        if prompt_len:
            seq = seq[prompt_len:] if len(seq) >= prompt_len else []

        # Find where generation likely started: assume after last occurrence of "Action sequence:" the model begins.
        # If you want cleaner, keep prompt without these special tokens so FSM sees none in prompt.
        # We'll just run FSM on full seq but it only reacts to our special tokens.

        # Extract only tokens that are in our relevant set + eos
        relevant = set([LBR, RBR, COL] + TOOL_IDS + VAL_DIST_IDS + VAL_TURN_IDS + [VAL_BARK_ID, eos])
        gen = [t for t in seq if t in relevant]

        if len(gen) == 0:
            # first constrained token should be "["
            return [LBR]

        # Determine FSM state by replaying the bracket structure on gen tokens.
        # We'll parse the last incomplete call boundary.
        # Count tokens since last completed "]".
        # Find last "]" in gen.
        last_rbr = -1
        for i in range(len(gen)-1, -1, -1):
            if gen[i] == RBR:
                last_rbr = i
                break
        tail = gen[last_rbr+1:] if last_rbr != -1 else gen

        # If last token is eos, nothing allowed (but HF may still call)
        if gen and gen[-1] == eos:
            return [eos]

        # If tail is empty => we are between calls: can start new call "[" or end EOS
        if len(tail) == 0:
            # allow EOS only if at least one full call exists
            has_one_call = (last_rbr != -1)
            return [LBR] + ([eos] if has_one_call else [])

        # Now interpret tail patterns:
        # tail[0] should be "[" if we're inside a call.
        if tail[0] != LBR:
            return [LBR]  # recover

        if len(tail) == 1:
            return TOOL_IDS

        tool_id = tail[1]
        if tool_id not in TOOL_IDS:
            return TOOL_IDS

        if len(tail) == 2:
            return [COL]

        if tail[2] != COL:
            return [COL]

        if len(tail) == 3:
            if tool_id in DIST_TOOL_IDS:
                return VAL_DIST_IDS
            elif tool_id in TURN_TOOL_IDS:
                return VAL_TURN_IDS
            else:
                return [VAL_BARK_ID]

        # value read
        if len(tail) == 4:
            return [RBR]

        # after "]" we should have started a new block (but tail includes only after last "]"; so shouldn't happen)
        return [LBR, eos]

    return allowed
