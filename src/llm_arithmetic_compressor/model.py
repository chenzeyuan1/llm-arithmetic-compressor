from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B"
DEFAULT_REVISION = "main"
DEFAULT_CONTEXT_WINDOW = 0


class ProbabilityModel(Protocol):
    model_name: str
    tokenizer_name: str
    revision: str
    vocab_size: int
    prefix_token_id: int
    max_context_tokens: int

    def encode_text(self, text: str) -> list[int]: ...

    def decode_tokens(self, token_ids: list[int]) -> str: ...

    def next_token_logits(self, context_token_ids: list[int]) -> Sequence[float]: ...

    def create_logit_session(self, context_window: int) -> "LogitSession": ...

    def runtime_info(self) -> dict: ...


class LogitSession(Protocol):
    def current_logits(self) -> Sequence[float]: ...

    def advance(self, token_id: int) -> None: ...


@dataclass(frozen=True)
class ModelConfig:
    model_name: str = DEFAULT_MODEL
    revision: str = DEFAULT_REVISION
    device: str = "auto"
    dtype: str = "float32"
    context_window: int = DEFAULT_CONTEXT_WINDOW


class HFModel:
    def __init__(self, config: ModelConfig) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Install torch and transformers to use the Hugging Face backend."
            ) from exc

        self.model_name = config.model_name
        self.tokenizer_name = config.model_name
        self.revision = config.revision
        self.device = _select_device(config.device, torch)
        self.dtype_name = config.dtype
        torch_dtype = _select_dtype(config.dtype, torch)

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            revision=config.revision,
            trust_remote_code=False,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            revision=config.revision,
            dtype=torch_dtype,
            trust_remote_code=False,
        )
        self.model.to(self.device)
        self.model.eval()
        self.torch = torch

        self.vocab_size = int(getattr(self.model.config, "vocab_size", len(self.tokenizer)))
        self.model_type = str(getattr(self.model.config, "model_type", "")).lower()
        prefix = self.tokenizer.bos_token_id
        if prefix is None:
            prefix = self.tokenizer.eos_token_id
        if prefix is None:
            raise RuntimeError("tokenizer has neither bos_token_id nor eos_token_id")
        self.prefix_token_id = int(prefix)
        if "mamba" in self.model_type:
            self.max_context_tokens = int(getattr(self.model.config, "max_position_embeddings", 1_000_000))
        else:
            self.max_context_tokens = int(getattr(self.model.config, "max_position_embeddings", 32768))

    def encode_text(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def decode_tokens(self, token_ids: list[int]) -> str:
        return str(
            self.tokenizer.decode(
                token_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        )

    def next_token_logits(self, context_token_ids: list[int]) -> Sequence[float]:
        if not context_token_ids:
            context_token_ids = [self.prefix_token_id]
        with self.torch.inference_mode():
            input_ids = self.torch.tensor([context_token_ids], dtype=self.torch.long, device=self.device)
            outputs = self.model(input_ids=input_ids, use_cache=False)
            logits = outputs.logits[0, -1, :].detach().to("cpu", dtype=self.torch.float32)
        return logits.numpy()

    def create_logit_session(self, context_window: int) -> "HFLogitSession":
        if "mamba" in self.model_type:
            return HFMambaLogitSession(self, context_window)
        return HFLogitSession(self, context_window)

    def runtime_info(self) -> dict:
        return {
            "device": str(self.device),
            "dtype": self.dtype_name,
            "model_type": self.model_type,
        }


class HFLogitSession:
    def __init__(self, hf_model: HFModel, context_window: int) -> None:
        self.hf_model = hf_model
        self.context_window = max(1, context_window)
        self.history: list[int] = []
        self.past_key_values = None
        self._logits: Sequence[float] | None = None
        self._rebuild_from_window()

    def current_logits(self) -> Sequence[float]:
        if self._logits is None:
            raise RuntimeError("logit session is not initialized")
        return self._logits

    def advance(self, token_id: int) -> None:
        self.history.append(token_id)
        if len(self.history) > self.context_window:
            self._rebuild_from_window()
            return

        torch = self.hf_model.torch
        with torch.inference_mode():
            input_ids = torch.tensor([[token_id]], dtype=torch.long, device=self.hf_model.device)
            outputs = self.hf_model.model(
                input_ids=input_ids,
                past_key_values=self.past_key_values,
                use_cache=True,
            )
            self.past_key_values = outputs.past_key_values
            self._logits = outputs.logits[0, -1, :].detach().to("cpu", dtype=torch.float32).numpy()

    def _rebuild_from_window(self) -> None:
        torch = self.hf_model.torch
        context = [self.hf_model.prefix_token_id] + self.history[-self.context_window :]
        with torch.inference_mode():
            input_ids = torch.tensor([context], dtype=torch.long, device=self.hf_model.device)
            outputs = self.hf_model.model(input_ids=input_ids, use_cache=True)
            self.past_key_values = outputs.past_key_values
            self._logits = outputs.logits[0, -1, :].detach().to("cpu", dtype=torch.float32).numpy()


class HFMambaLogitSession:
    def __init__(self, hf_model: HFModel, context_window: int) -> None:
        self.hf_model = hf_model
        self.context_window = max(1, context_window)
        self.history: list[int] = []
        self.cache_params = None
        self._logits: Sequence[float] | None = None
        self._rebuild_from_window()

    def current_logits(self) -> Sequence[float]:
        if self._logits is None:
            raise RuntimeError("logit session is not initialized")
        return self._logits

    def advance(self, token_id: int) -> None:
        self.history.append(token_id)
        if len(self.history) > self.context_window:
            self._rebuild_from_window()
            return

        torch = self.hf_model.torch
        with torch.inference_mode():
            input_ids = torch.tensor([[token_id]], dtype=torch.long, device=self.hf_model.device)
            outputs = self.hf_model.model(
                input_ids=input_ids,
                cache_params=self.cache_params,
                use_cache=True,
            )
            self.cache_params = outputs.cache_params
            self._logits = outputs.logits[0, -1, :].detach().to("cpu", dtype=torch.float32).numpy()

    def _rebuild_from_window(self) -> None:
        torch = self.hf_model.torch
        context = [self.hf_model.prefix_token_id] + self.history[-self.context_window :]
        with torch.inference_mode():
            input_ids = torch.tensor([context], dtype=torch.long, device=self.hf_model.device)
            outputs = self.hf_model.model(input_ids=input_ids, use_cache=True)
            self.cache_params = outputs.cache_params
            self._logits = outputs.logits[0, -1, :].detach().to("cpu", dtype=torch.float32).numpy()


class FakeByteModel:
    """Small deterministic model for tests; tokens are Unicode code points."""

    model_name = "fake-byte-model"
    tokenizer_name = "fake-byte-tokenizer"
    revision = "test"
    vocab_size = 256
    prefix_token_id = 0
    max_context_tokens = 256

    def encode_text(self, text: str) -> list[int]:
        data = text.encode("utf-8")
        return list(data)

    def decode_tokens(self, token_ids: list[int]) -> str:
        return bytes(token_ids).decode("utf-8")

    def next_token_logits(self, context_token_ids: list[int]) -> list[float]:
        logits = [-4.0] * self.vocab_size
        if context_token_ids:
            logits[context_token_ids[-1] % self.vocab_size] = 3.0
        logits[32] = max(logits[32], 1.0)
        return logits

    def create_logit_session(self, context_window: int) -> "FakeLogitSession":
        return FakeLogitSession(self, context_window)

    def runtime_info(self) -> dict:
        return {
            "device": "fake",
            "dtype": "float32",
            "model_type": "",
        }


class FakeLogitSession:
    def __init__(self, model: FakeByteModel, context_window: int) -> None:
        self.model = model
        self.context_window = max(1, context_window)
        self.history: list[int] = []
        self.calls = 0

    def current_logits(self) -> list[float]:
        start = max(0, len(self.history) - self.context_window)
        context = [self.model.prefix_token_id] + self.history[start:]
        self.calls += 1
        return self.model.next_token_logits(context)

    def advance(self, token_id: int) -> None:
        self.history.append(token_id)


def load_model(config: ModelConfig) -> ProbabilityModel:
    return HFModel(config)


def _select_device(device: str, torch_module):
    if device != "auto":
        return torch_module.device(device)
    if getattr(torch_module.backends, "mps", None) and torch_module.backends.mps.is_available():
        return torch_module.device("mps")
    if torch_module.cuda.is_available():
        return torch_module.device("cuda")
    return torch_module.device("cpu")


def _select_dtype(dtype: str, torch_module):
    if dtype == "float32":
        return torch_module.float32
    if dtype == "float16":
        return torch_module.float16
    if dtype == "bfloat16":
        return torch_module.bfloat16
    raise ValueError(f"unsupported dtype: {dtype}")
