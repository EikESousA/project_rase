from typing import Any

from unsloth import FastLanguageModel

from app.configs.model_config import ModelConfig


def model_unsloth(model: str) -> tuple[Any, Any]:
    """Digitar"""

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model,
        max_seq_length=ModelConfig.MAX_SEQ_LENGTH,
        dtype=ModelConfig.DTYPE,
        load_in_4bit=ModelConfig.LOAD_IN_4BIT,
    )

    return model, tokenizer
