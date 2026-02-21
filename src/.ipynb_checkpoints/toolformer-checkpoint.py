import re

from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

CALL_RE = re.compile(r'\[(forward|backward|left|right):(\d+)\]')

def execute_calls(text):
    def replace(match):
        name, arg = match.group(1), match.group(2)
        result = tools[name](arg)
        return f'[{name}({arg}) → {result}]'
    return CALL_RE.sub(replace, text)

tools = {
    "forward":  lambda dist: "FORWARD: "  + str(dist),
    "backward": lambda dist: "BACKWARD: " + str(dist),
    "left":     lambda dist: "LEFT: "     + str(dist),
    "right":    lambda dist: "RIGHT: "    + str(dist),
}

def helpful(original, augmented, model, tokenizer, threshold=1.0):
    loss_without = compute_loss(original, model, tokenizer)
    loss_with = compute_loss(augmented model, tokenizer)

    return ( (loss_without - loss_with) > threshold)


trainer = Trainer(
    model=model,
    train_dataset=augmented_dataset,
    args=TrainingArguments(...)
)
trainer.train()

def generate_with_tools(prompt, model, tokenizer, max_new_tokens=200):
    tokens = tokenizer(prompt, return_tensors="pt").input_ids
    output = ""
    while len(output) < max_new_tokens:
        next_token = model.generate(tokens, max_new_tokens=1)
        output += tokenizer.decode(next_token[0, -1])
        # Check if we just completed a tool call input
        match = re.search(r'\[(\w+)\((.+?)\)(?! →)', output)
        if match:
            name, arg = match.group(1), match.group(2)
            result = tools[name](arg)
            output += f" → {result}]"
            tokens = tokenizer(prompt + output, return_tensors="pt").input_ids
    return output