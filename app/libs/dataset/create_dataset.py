from typing import Any

from datasets import Dataset, load_dataset

from app.libs.prompt.format_prompt import format_prompt


def create_dataset(path: str, prompt: str, eos_token: Any) -> Dataset:
    """Digitar"""

    dataset = load_dataset("json", data_files=path, split="train")
    dataset = dataset.map(
        lambda examples: format_prompt(examples=examples, prompt=prompt, eos_token=eos_token), batched=True
    )

    return dataset
