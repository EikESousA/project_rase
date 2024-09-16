import sys

from app.configs.dataset_config import DatasetConfig
from app.configs.model_config import ModelConfig
from app.libs.dataset.create_dataset import create_dataset
from app.libs.memory.consumer_memory import consumer_memory
from app.libs.memory.reserved_memory import reserved_memory
from app.libs.prompt.alpaca_prompt import alpaca_prompt
from app.libs.save.file_save import file_save
from app.libs.trl.train_trl import train_trl
from app.libs.trl.trainer_trl import trainer_trl
from app.libs.unsloth.inference_unsloth import inference_unsloth
from app.libs.unsloth.model_unsloth import model_unsloth
from app.libs.unsloth.parameters_unsloth import parameters_unsloth
from app.utils.log.print_log import print_log


def instruction_service() -> None:
    """Digitar"""

    try:

        model, tokenizer = model_unsloth(model=ModelConfig.MODEL_LLAMA)

        model = parameters_unsloth(model)

        prompt = alpaca_prompt()

        dataset = create_dataset(path=DatasetConfig.CABRITA_PATH, prompt=prompt, eos_token=tokenizer.eos_token)

        trainer = trainer_trl(model=model, tokenizer=tokenizer, dataset=dataset)

        start_gpu_memory, max_memory = reserved_memory()

        trainer_stats = train_trl(trainer=trainer)

        consumer_memory(start_gpu_memory=start_gpu_memory, max_memory=max_memory, trainer_stats=trainer_stats)

        inference_unsloth(
            model=model, tokenizer=tokenizer, prompt=prompt, question="Qual é a diferença entre uma rede e a internet?"
        )

        file_save(model=model, tokenizer=tokenizer)

    except KeyboardInterrupt:
        print_log(finish=False, module="SYST", action="EXIT", message="Encerrado.")

        sys.exit(0)
