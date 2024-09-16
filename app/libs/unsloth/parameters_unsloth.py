from typing import Any

from unsloth import FastLanguageModel

from app.configs.finetunning_config import FineTunningConfig


def parameters_unsloth(model: Any) -> Any:
    """Digitar"""

    model = FastLanguageModel.get_peft_model(
        model,
        r=FineTunningConfig.R,
        target_modules=FineTunningConfig.TARGET_MODULES,
        lora_alpha=FineTunningConfig.LORA_ALPHA,
        lora_dropout=FineTunningConfig.LORA_DROPOUT,
        bias=FineTunningConfig.BIAS,
        use_gradient_checkpointing=FineTunningConfig.USE_GRADIENT_CHECKPOINTING,
        random_state=FineTunningConfig.RANDOM_STATE,
        use_rslora=FineTunningConfig.USE_RSLORA,
        loftq_config=FineTunningConfig.LOFTQ_CONFIG,
    )

    return model
