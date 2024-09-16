from typing import Any


def huggingface_save(model: Any, tokenizer: Any) -> None:
    """Digitar"""

    model.push_to_hub_gguf(
        "",
        tokenizer,
        quantization_method=[
            "q4_k_m",
            "q8_0",
            "q5_k_m",
        ],
        token="",
    )
