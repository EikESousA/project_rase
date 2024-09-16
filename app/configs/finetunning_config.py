from typing import Optional


class FineTunningConfig:
    R = 16
    TARGET_MODULES = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    LORA_ALPHA = 16
    LORA_DROPOUT = 0
    BIAS = "none"
    USE_GRADIENT_CHECKPOINTING = "unsloth"
    RANDOM_STATE = 3407
    USE_RSLORA = False
    LOFTQ_CONFIG: Optional[str] = None
