from typing import Any, Optional

from app.configs.model_config import ModelConfig


def file_save(model: Any, tokenizer: Any, quantization_method: Optional[str] = "q4_k_m") -> None:
    """Digitar"""

    model.save_pretrained_gguf(ModelConfig.MODEL_TRAIN, tokenizer, quantization_method=quantization_method)
