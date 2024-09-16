from typing import Any

from transformers import TextStreamer
from unsloth import FastLanguageModel


def inference_unsloth(model: Any, tokenizer: Any, prompt: str, question: str) -> None:
    """Digitar"""

    FastLanguageModel.for_inference(model)

    inputs = tokenizer(
        [
            prompt.format(
                "Você é um assistente do serviço de atendimento ao cliente que deve responder perguntas dos clientes",
                question,
                "",
            )
        ],
        return_tensors="pt",
    ).to("cuda")

    text_streamer = TextStreamer(tokenizer)
    _ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=128)
