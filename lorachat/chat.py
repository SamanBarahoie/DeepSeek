
import torch

from transformers import PreTrainedTokenizerFast
from collections import OrderedDict
import logging

# function to build the base model from config
from typing import Optional
from models.deepseek import Transformer
from configs.model_config import TextConfig

def build_model(cfg: TextConfig, device: str) -> Transformer:
    model = Transformer(cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(
        "Model initialized with %d parameters (~%.2fM)",
        total_params,
        total_params / 1e6,
    )
    return model


class LoRAChatModel:
    def __init__(
        self,
        model_path: str,  # Path to fine-tuned checkpoint or full model
        tokenizer_path: str,  # Path to tokenizer
        config: dict,  # General config (e.g., seq_len)
        model_class=None,  # If checkpoint is state_dict, pass model class
        model_init_kwargs: dict = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the LoRA fine-tuned model and tokenizer for chatting.

        Args:
            model_path (str): Path to the fine-tuned model checkpoint or full model.
            tokenizer_path (str): Path to the tokenizer.
            config (dict): General configuration with parameters like seq_len. Should include 'model_cfg' for build_model.
            model_class (nn.Module, optional): Class to instantiate model when loading state_dict.
            model_init_kwargs (dict, optional): kwargs to instantiate model_class.
            device (str): Device to run the model on ('cuda' or 'cpu').
        """
        self.device = device
        self.config = config
        self.seq_len = 192
        # Load tokenizer
        try:
            self.tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
        except Exception as e:
            raise ValueError(f"Failed to load tokenizer from {tokenizer_path}: {e}")

        # Load model or state_dict
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            # If checkpoint is state_dict (OrderedDict)
            if isinstance(checkpoint, OrderedDict):
                # if user provided a build_model config, use build_model
                if model_class is None:
                    # require config['model_cfg'] exists for build_model
                    if 'model_cfg' not in self.config:
                        raise ValueError("No model_class provided and 'model_cfg' not found in config for build_model.")
                    model = build_model(self.config['model_cfg'], self.device)
                else:
                    model = model_class(**(model_init_kwargs or {})).to(self.device)
                model.load_state_dict(checkpoint)
            else:
                model = checkpoint
            model.to(self.device)
            model.eval()
            self.model = model
        except Exception as e:
            raise ValueError(f"Failed to initialize or load model from {model_path}: {e}")



    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        top_k: int = 50,
        temperature: float = 0.7,
    ) -> str:
        # Enable LoRA in attention layers
        for i, layer in enumerate(self.model.layers):
            if hasattr(layer.attention, "enable_lora"):
                layer.attention.enable_lora()
                print(f"LoRA enabled on layer {i}")  # برای تأیید فعال شدن
        self.model.eval()
        for name, param in self.model.named_parameters():
            if "lora" in name:
                print(f"{name} - requires_grad: {param.requires_grad}")

        with torch.no_grad():
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.seq_len,
            ).input_ids.to(self.device)

            if inputs.max().item() >= self.tokenizer.vocab_size:
                raise ValueError(
                    f"Input contains invalid token ID {inputs.max().item()} >= vocab_size {self.tokenizer.vocab_size}"
                )

            generated = inputs.clone()
            past = None


            for step in range(max_new_tokens):
                input_ids = generated if past is None else generated[:, -1:]

                # Model may return tuple: (logits, past_key_values, ...)
                out = self.model(
                    input_ids=input_ids,
                    attention_mask=(input_ids != self.tokenizer.pad_token_id).float(),
                    past_key_values=past,
                    use_cache=True,
                    output_attentions=False,
                    output_router_logits=False,
                )
                # unpack
                if isinstance(out, tuple):
                    logits = out[0]
                    past = out[1]
                else:
                    logits = out.logits
                    past = out.past_key_values

                logits = logits[:, -1, :] / temperature
                top_k_probs, top_k_idx = torch.topk(logits, top_k, dim=-1)
                probs = torch.softmax(top_k_probs, dim=-1)
                sample_idx = torch.multinomial(probs, num_samples=1)
                next_token = top_k_idx.gather(-1, sample_idx)

                generated = torch.cat([generated, next_token], dim=-1)

                if next_token.item() == self.tokenizer.eos_token_id:
                    break

            return self.tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)


    def chat(self, prompt: str, max_new_tokens: int = 50, top_k: int = 50, temperature: float = 0.7) -> str:
        """
        Interactive chat method to generate a response to a user prompt.
        """
        try:
            return self.generate_text(prompt, max_new_tokens, top_k, temperature)
        except Exception as e:
            return f"Error generating response: {e}"


