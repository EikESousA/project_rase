from typing import Any

from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import is_bfloat16_supported

from app.configs.model_config import ModelConfig
from app.configs.trainning_config import TrainningConfig


def trainer_trl(model: Any, tokenizer: Any, dataset: Dataset) -> SFTTrainer:
    """Digitar"""

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=ModelConfig.MAX_SEQ_LENGTH,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=TrainningConfig.PER_DEVICE_TRAIN_BATCH_SIZE,
            gradient_accumulation_steps=TrainningConfig.GRADIENT_ACCUMULATION_STEPS,
            warmup_steps=5,
            num_train_epochs=TrainningConfig.NUM_TRAIN_EPOCHS,
            # max_steps = max_steps,
            learning_rate=TrainningConfig.LEARNING_RATE,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
        ),
    )

    return trainer
