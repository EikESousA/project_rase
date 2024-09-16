from typing import Any

from trl import SFTTrainer


def train_trl(trainer: SFTTrainer) -> Any:
    """Digitar"""

    trainer_stats = trainer.train()

    return trainer_stats
