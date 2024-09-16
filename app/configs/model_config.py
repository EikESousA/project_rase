from typing import Optional


class ModelConfig:
    MAX_SEQ_LENGTH = 2048
    DTYPE: Optional[str] = None
    LOAD_IN_4BIT = True

    MODEL_LLAMA = "unsloth/Meta-Llama-3.1-8B-Instruct"
    MODEL_TRAIN = ""
