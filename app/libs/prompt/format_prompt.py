def format_prompt(examples, prompt, eos_token):
    """Digitar"""

    inss = examples["instruction"]
    inps = examples["input"]
    outs = examples["output"]
    texts = []

    for ins, inp, out in zip(inss, inps, outs):
        text = prompt.format(ins, inp, out) + eos_token
        texts.append(text)

    return {"text": texts}
