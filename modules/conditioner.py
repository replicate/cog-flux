import os

import torch
from torch import Tensor, nn
from transformers import (
    CLIPTextModel,
    CLIPTokenizer,
    T5EncoderModel,
    T5Tokenizer,
    __version__,
)
from transformers.utils.quantization_config import QuantoConfig, BitsAndBytesConfig

CACHE_DIR = os.environ.get("HF_HOME", "~/.cache/huggingface")


def auto_quantization_config(
    quantization_dtype: str,
) -> QuantoConfig | BitsAndBytesConfig:
    if quantization_dtype == "qfloat8":
        return QuantoConfig(weights="float8")
    elif quantization_dtype == "qint4":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
    elif quantization_dtype == "qint8":
        return BitsAndBytesConfig(load_in_8bit=True, llm_int8_has_fp16_weight=False)
    elif quantization_dtype == "qint2":
        return QuantoConfig(weights="int2")
    else:
        raise ValueError(f"Unsupported quantization dtype: {quantization_dtype}")


class HFEmbedder(nn.Module):
    def __init__(
        self,
        version: str,
        max_length: int,
        device: torch.device | int,
        quantization_dtype: str | None = None,
        offloading_device: torch.device | int | None = torch.device("cpu"),
        **hf_kwargs,
    ):
        super().__init__()
        self.offloading_device = (
            offloading_device
            if isinstance(offloading_device, torch.device)
            else torch.device(offloading_device)
        )
        self.device = (
            device if isinstance(device, torch.device) else torch.device(device)
        )
        self.is_clip = version.startswith("openai")
        self.max_length = max_length
        self.output_key = "pooler_output" if self.is_clip else "last_hidden_state"

        auto_quant_config = (
            auto_quantization_config(quantization_dtype) if quantization_dtype else None
        )

        # BNB will move to cuda:0 by default if not specified
        if isinstance(auto_quant_config, BitsAndBytesConfig):
            hf_kwargs["device_map"] = {"": self.device.index}
        if auto_quant_config is not None:
            hf_kwargs["quantization_config"] = auto_quant_config

        if self.is_clip:
            self.tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(
                version, max_length=max_length
            )

            self.hf_module: CLIPTextModel = CLIPTextModel.from_pretrained(
                version,
                **hf_kwargs,
            )

        else:
            self.tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(
                os.path.join(version, 'tokenizer'), max_length=max_length
            )
            self.hf_module: T5EncoderModel = T5EncoderModel.from_pretrained(
                os.path.join(version, 'model'),
                **hf_kwargs,
            )

    def offload(self):
        self.hf_module.to(device=self.offloading_device)
        torch.cuda.empty_cache()

    def cuda(self):
        self.hf_module.to(device=self.device)

    def forward(self, text: list[str]) -> Tensor:
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=False,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        outputs = self.hf_module(
            input_ids=batch_encoding["input_ids"].to(self.hf_module.device),
            attention_mask=None,
            output_hidden_states=False,
        )
        return outputs[self.output_key]


if __name__ == "__main__":
    model = HFEmbedder(
        "city96/t5-v1_1-xxl-encoder-bf16",
        max_length=512,
        device=0,
        quantization_dtype="qfloat8",
    )
    o = model(["hello"])
    print(o)
