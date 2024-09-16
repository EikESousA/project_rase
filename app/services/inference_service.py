import sys

from app.configs.model_config import ModelConfig
from app.libs.prompt.alpaca_prompt import alpaca_prompt
from app.libs.unsloth.inference_unsloth import inference_unsloth
from app.libs.unsloth.model_unsloth import model_unsloth
from app.libs.unsloth.parameters_unsloth import parameters_unsloth
from app.utils.log.print_log import print_log


def inference_service() -> None:
    """Digitar"""

    try:
        model, tokenizer = model_unsloth(model=ModelConfig.MODEL_TRAIN)

        model = parameters_unsloth(model)

        prompt = alpaca_prompt()

        inference_unsloth(
            model=model, tokenizer=tokenizer, prompt=prompt, question="Qual é a diferença entre uma rede e a internet?"
        )

    except KeyboardInterrupt:
        print_log(finish=False, module="SYST", action="EXIT", message="Encerrado.")

        sys.exit(0)
