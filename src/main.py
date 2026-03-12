from __future__ import annotations

import base64
import io
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from urllib.parse import unquote, urlparse
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = Path(os.getenv("MODEL_DIR", PROJECT_ROOT / "models")).resolve()
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Configure cache locations before importing libraries that may resolve them at import time.
os.environ["HF_HOME"] = str(MODEL_DIR)
os.environ["HF_HUB_CACHE"] = str(MODEL_DIR / "hub")
os.environ["HUGGINGFACE_HUB_CACHE"] = str(MODEL_DIR / "hub")
os.environ["TRANSFORMERS_CACHE"] = str(MODEL_DIR / "transformers")
os.environ["HF_ASSETS_CACHE"] = str(MODEL_DIR / "assets")

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import mlx.core as mx
from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template
from PIL import Image
from pydantic import BaseModel, Field
import requests


def _patch_transformers_video_processor_none_bug() -> None:
    try:
        import importlib
        from transformers.models.auto import video_processing_auto
    except Exception:
        return

    def safe_video_processor_class_from_name(class_name: str | None):
        if class_name is None:
            return None

        mapping_names = video_processing_auto.VIDEO_PROCESSOR_MAPPING_NAMES
        extra_content = video_processing_auto.VIDEO_PROCESSOR_MAPPING._extra_content.values()
        model_type_to_module_name = video_processing_auto.model_type_to_module_name

        for module_name, extractors in mapping_names.items():
            if not extractors:
                continue
            if class_name in extractors:
                module_name = model_type_to_module_name(module_name)
                module = importlib.import_module(f".{module_name}", "transformers.models")
                try:
                    return getattr(module, class_name)
                except AttributeError:
                    continue

        for extractor in extra_content:
            if getattr(extractor, "__name__", None) == class_name:
                return extractor

        main_module = importlib.import_module("transformers")
        if hasattr(main_module, class_name):
            return getattr(main_module, class_name)

        return None

    video_processing_auto.video_processor_class_from_name = safe_video_processor_class_from_name


_patch_transformers_video_processor_none_bug()


MODEL_ID = os.getenv("MODEL_ID", "mlx-community/Qwen3.5-9B-4bit")
TRUST_REMOTE_CODE = os.getenv("TRUST_REMOTE_CODE", "false").lower() in {"1", "true", "yes", "on"}
DEFAULT_MAX_TOKENS = int(os.getenv("DEFAULT_MAX_TOKENS", "512"))
MAX_CONTEXT_LENGTH = int(os.getenv("MAX_CONTEXT_LENGTH", "32768"))
LM_STUDIO_DEFAULT_MAX_TOKENS = int(os.getenv("LM_STUDIO_DEFAULT_MAX_TOKENS", "128"))
STRIP_THINKING_DEFAULT = os.getenv("STRIP_THINKING_DEFAULT", "false").lower() in {"1", "true", "yes", "on"}
PRELOAD_MODEL_ON_STARTUP = os.getenv("PRELOAD_MODEL_ON_STARTUP", "true").lower() in {"1", "true", "yes", "on"}

BOOT_STARTED_AT = time.perf_counter()


def _log(message: str) -> None:
    print(f"[sample-mlx-vlm] {message}", flush=True)


class ChatMessage(BaseModel):
    role: str
    content: str | list[dict[str, Any]]


class ChatCompletionsRequest(BaseModel):
    model: str = Field(default=MODEL_ID)
    messages: list[ChatMessage]
    max_tokens: int = Field(default=DEFAULT_MAX_TOKENS, ge=1, le=81920)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    repetition_penalty: float = Field(default=1.0, ge=0.1, le=3.0)
    strip_thinking: bool | None = None
    extra_body: dict[str, Any] | None = None


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = Field(default=DEFAULT_MAX_TOKENS, ge=1, le=81920)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    repetition_penalty: float = Field(default=1.0, ge=0.1, le=3.0)
    strip_thinking: bool | None = None


class LmStudioChatInputItem(BaseModel):
    type: str
    content: str | None = None
    data_url: str | None = None
    url: str | None = None
    audio: str | None = None
    input_audio: str | None = None


class LmStudioChatRequest(BaseModel):
    model: str | None = None
    system_prompt: str | None = None
    input: str | list[LmStudioChatInputItem]
    context_length: int | None = None
    max_tokens: int = Field(default=LM_STUDIO_DEFAULT_MAX_TOKENS, ge=1, le=81920)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    repetition_penalty: float = Field(default=1.0, ge=0.1, le=3.0)
    strip_thinking: bool | None = None


class ModelRuntime:
    def __init__(self) -> None:
        self._model = None
        self._processor = None
        self._load_error: str | None = None
        self._lock = Lock()

    @property
    def model(self):
        self._ensure_loaded()
        return self._model

    @property
    def processor(self):
        self._ensure_loaded()
        return self._processor

    def _ensure_loaded(self) -> None:
        with self._lock:
            if self._model is not None and self._processor is not None:
                return
            if self._load_error is not None:
                raise RuntimeError(self._load_error)
            try:
                self._model, self._processor = load(MODEL_ID, trust_remote_code=TRUST_REMOTE_CODE)
            except Exception as exc:
                self._load_error = (
                    f"failed to load model '{MODEL_ID}': {exc}. "
                    "If the model requires custom Hugging Face processor code, set TRUST_REMOTE_CODE=true. "
                    "This environment may also need additional processor dependencies; try Python 3.12/3.13 and install mlx-vlm with torch extras (pip install -U 'mlx-vlm[torch]')."
                )
                raise RuntimeError(self._load_error) from exc


runtime = ModelRuntime()
_log(
    f"starting server: model={MODEL_ID}, model_dir={MODEL_DIR}, trust_remote_code={'on' if TRUST_REMOTE_CODE else 'off'}, preload={'on' if PRELOAD_MODEL_ON_STARTUP else 'off'}, strip_thinking_default={'on' if STRIP_THINKING_DEFAULT else 'off'}"
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _log("initializing application")
    try:
        if PRELOAD_MODEL_ON_STARTUP:
            _log(f"loading model: {MODEL_ID}")
            runtime.model
            runtime.processor
            _log("model loaded")
        else:
            _log("model preload disabled")

        elapsed = time.perf_counter() - BOOT_STARTED_AT
        _log(f"startup complete in {elapsed:.2f}s")
        yield
    except Exception as exc:
        _log(f"startup failed: {exc}")
        raise
    finally:
        _log("shutdown complete")


app = FastAPI(title="Local LLM Server", version="0.2.0", lifespan=lifespan)


def _validate_model_id(request_model: str | None) -> None:
    if request_model is None:
        return
    if request_model != MODEL_ID:
        raise HTTPException(status_code=400, detail=f"unsupported model: {request_model}. available: {MODEL_ID}")


def _decode_data_url(data_url: str) -> tuple[str, bytes]:
    if not data_url.startswith("data:") or "," not in data_url:
        raise ValueError("invalid data_url")
    header, payload = data_url.split(",", 1)
    mime = header[5:].split(";", 1)[0] if header.startswith("data:") else "application/octet-stream"
    if ";base64" in header:
        return mime, base64.b64decode(payload)
    return mime, payload.encode("utf-8")


def _image_from_data_url(data_url: str) -> Image.Image:
    mime, raw = _decode_data_url(data_url)
    if not mime.startswith("image/"):
        raise ValueError("data_url must be image/*")
    image = Image.open(io.BytesIO(raw))
    return image.convert("RGB")


def _image_from_url(image_url: str) -> Image.Image:
    parsed = urlparse(image_url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("image_url must be a data URL or http(s) URL")

    try:
        response = requests.get(image_url, timeout=20)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise ValueError(f"failed to download image_url: {exc}") from exc

    try:
        image = Image.open(io.BytesIO(response.content))
    except Exception as exc:
        raise ValueError(f"failed to decode image_url content: {exc}") from exc

    return image.convert("RGB")


def _image_from_source(source: str) -> Image.Image:
    if source.startswith("data:"):
        return _image_from_data_url(source)
    return _image_from_url(source)


def _video_source_from_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme in {"http", "https"}:
        return url
    if parsed.scheme == "file":
        local_path = unquote(parsed.path or "")
        if not local_path:
            raise ValueError("video_url file:// must include a local path")
        return local_path
    raise ValueError("video_url must be a file/http(s) URL")


def _video_source_from_data_url(data_url: str) -> str:
    raise ValueError("video data_url is not supported. use video_url/url with http(s) or file://")


def _video_source_from_input(source: str) -> str:
    if source.startswith("data:"):
        return _video_source_from_data_url(source)
    return _video_source_from_url(source)


def _audio_source_from_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme in {"http", "https"}:
        return url
    if parsed.scheme == "file":
        local_path = unquote(parsed.path or "")
        if not local_path:
            raise ValueError("audio file:// must include a local path")
        return local_path
    if parsed.scheme == "":
        return url
    raise ValueError("audio input must be a local path, file:// URL, or http(s) URL")


def _audio_source_from_data_url(data_url: str) -> str:
    raise ValueError("audio data_url is not supported. use a local path, file:// URL, or http(s) URL")


def _audio_source_from_input(source: str) -> str:
    if source.startswith("data:"):
        return _audio_source_from_data_url(source)
    return _audio_source_from_url(source)


def _extract_text(content: str | list[dict[str, Any]]) -> str:
    if isinstance(content, str):
        return content
    texts: list[str] = []
    for item in content:
        if isinstance(item, dict) and item.get("type") in {"text", "input_text"} and isinstance(item.get("text"), str):
            texts.append(item["text"])
        elif isinstance(item, dict) and item.get("type") == "message" and isinstance(item.get("content"), str):
            texts.append(item["content"])
    return "\n".join(texts)


def _extract_openai_input(messages: list[ChatMessage]) -> tuple[list[dict[str, Any]], list[Image.Image], list[str], list[str]]:
    converted: list[dict[str, Any]] = []
    images: list[Image.Image] = []
    videos: list[str] = []
    audios: list[str] = []

    for message in messages:
        if isinstance(message.content, str):
            converted.append({"role": message.role, "content": message.content})
            continue

        text_parts: list[str] = []
        for item in message.content:
            if not isinstance(item, dict):
                continue
            item_type = (item.get("type") or "").lower()

            if item_type in {"text", "input_text"} and isinstance(item.get("text"), str):
                text_parts.append(item["text"])
                continue

            if item_type == "message" and isinstance(item.get("content"), str):
                text_parts.append(item["content"])
                continue

            if item_type in {"image_url", "image", "input_image"}:
                image_url = item.get("image_url")
                url: str | None = None
                if isinstance(image_url, dict) and isinstance(image_url.get("url"), str):
                    url = image_url["url"]
                elif isinstance(image_url, str):
                    url = image_url
                elif isinstance(item.get("url"), str):
                    url = item["url"]
                if isinstance(url, str) and url.strip():
                    images.append(_image_from_source(url.strip()))
                elif url is not None:
                    raise ValueError("image_url must be a data URL (data:image/...;base64,...) or http(s) URL")
                continue

            if item_type in {"video", "video_url"}:
                video_url = item.get("video_url")
                url = None
                if isinstance(video_url, dict) and isinstance(video_url.get("url"), str):
                    url = video_url["url"]
                elif isinstance(item.get("url"), str):
                    url = item["url"]
                if isinstance(url, str) and url.strip():
                    source = _video_source_from_input(url.strip())
                    videos.append(source)
                elif url is not None:
                    raise ValueError("video_url must be a file/http(s) URL")
                continue

            if item_type in {"audio", "audio_url", "input_audio"}:
                audio_input = item.get("input_audio")
                audio_url = item.get("audio_url")
                source = None
                if isinstance(audio_input, str):
                    source = audio_input
                elif isinstance(audio_input, dict) and isinstance(audio_input.get("url"), str):
                    source = audio_input["url"]
                elif isinstance(audio_url, str):
                    source = audio_url
                elif isinstance(audio_url, dict) and isinstance(audio_url.get("url"), str):
                    source = audio_url["url"]
                elif isinstance(item.get("audio"), str):
                    source = item["audio"]
                elif isinstance(item.get("url"), str):
                    source = item["url"]

                if isinstance(source, str) and source.strip():
                    audios.append(_audio_source_from_input(source.strip()))
                elif isinstance(audio_input, dict) and isinstance(audio_input.get("data"), str):
                    audios.append(_audio_source_from_input(audio_input["data"].strip()))
                elif source is not None or audio_input is not None or audio_url is not None:
                    raise ValueError("input_audio must be a local path, file:// URL, or http(s) URL")
                continue

        converted.append({"role": message.role, "content": "\n".join(text_parts).strip()})

    return converted, images, videos, audios


def _extract_lm_studio_input(
    input_data: str | list[LmStudioChatInputItem],
) -> tuple[list[dict[str, Any]], list[Image.Image], list[str], list[str]]:
    if isinstance(input_data, str):
        content = [{"type": "text", "text": input_data}] if input_data.strip() else []
        return content, [], [], []

    content_parts: list[dict[str, Any]] = []
    images: list[Image.Image] = []
    videos: list[str] = []
    audios: list[str] = []

    for item in input_data:
        item_type = (item.type or "").lower()
        if item_type in {"message", "text"} and isinstance(item.content, str):
            text = item.content.strip()
            if text:
                content_parts.append({"type": "text", "text": text})
        elif item_type == "image":
            source = None
            if isinstance(item.data_url, str) and item.data_url.strip():
                source = item.data_url.strip()
            elif isinstance(item.url, str) and item.url.strip():
                source = item.url.strip()

            if source is not None:
                images.append(_image_from_source(source))
            else:
                raise ValueError("image input requires data_url (data:image/...;base64,...) or url (http(s)://...)")
        elif item_type == "video":
            source = None
            if isinstance(item.url, str) and item.url.strip():
                source = item.url.strip()
            elif isinstance(item.data_url, str) and item.data_url.strip():
                source = item.data_url.strip()

            if source is not None:
                video_source = _video_source_from_input(source)
                videos.append(video_source)
            else:
                raise ValueError("video input requires url (file/http(s)://...)")

        elif item_type in {"audio", "input_audio"}:
            source = None
            if isinstance(item.input_audio, str) and item.input_audio.strip():
                source = item.input_audio.strip()
            elif isinstance(item.audio, str) and item.audio.strip():
                source = item.audio.strip()
            elif isinstance(item.url, str) and item.url.strip():
                source = item.url.strip()
            elif isinstance(item.data_url, str) and item.data_url.strip():
                source = item.data_url.strip()

            if source is not None:
                audios.append(_audio_source_from_input(source))
            else:
                raise ValueError("audio input requires a local path, file:// URL, or http(s) URL")

    return content_parts, images, videos, audios


def _safe_apply_chat_template(
    messages: list[dict[str, Any]],
    image_count: int,
    video_count: int,
    audio_count: int,
    disable_thinking: bool = False,
    video_source: str | None = None,
) -> str:
    processor = runtime.processor
    config = getattr(runtime.model, "config", None)
    model_type = getattr(config, "model_type", "")

    if video_count > 0 and hasattr(processor, "apply_chat_template"):
        video_messages: list[dict[str, Any]] = []
        media_attached = False
        target_index = -1
        for index in range(len(messages) - 1, -1, -1):
            if messages[index].get("role") == "user":
                target_index = index
                break
        if target_index == -1 and messages:
            target_index = len(messages) - 1

        for index, message in enumerate(messages):
            role = str(message.get("role", "user"))
            content = message.get("content", "")
            text = content if isinstance(content, str) else ""

            if index == target_index and not media_attached and video_source is not None:
                parts: list[dict[str, Any]] = [
                    {
                        "type": "video",
                        "video": video_source,
                        "max_pixels": 224 * 224,
                        "fps": 1,
                    }
                ]
                if text:
                    parts.append({"type": "text", "text": text})
                video_messages.append({"role": role, "content": parts})
                media_attached = True
                continue

            video_messages.append({"role": role, "content": text})

        return processor.apply_chat_template(
            video_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    if model_type == "qwen3_omni_moe" and hasattr(processor, "apply_chat_template"):
        return processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    if image_count > 0 or video_count > 0 or audio_count > 0:
        try:
            return apply_chat_template(
                processor,
                config,
                messages,
                num_images=image_count,
                num_audios=audio_count,
                enable_thinking=not disable_thinking,
                video=video_source,
            )
        except TypeError:
            return apply_chat_template(processor, config, messages, video=video_source)

    if hasattr(processor, "apply_chat_template"):
        try:
            return processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                num_images=image_count,
                num_audios=audio_count,
                enable_thinking=not disable_thinking,
                video=video_source,
            )
        except TypeError:
            try:
                return processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    num_images=image_count,
                    num_audios=audio_count,
                    video=video_source,
                )
            except Exception:
                pass
        except Exception:
            pass
    try:
        return apply_chat_template(
            processor,
            config,
            messages,
            num_images=image_count,
            num_audios=audio_count,
            enable_thinking=not disable_thinking,
            video=video_source,
        )
    except TypeError:
        return apply_chat_template(processor, config, messages, video=video_source)


def _estimate_tokens(text: str) -> int:
    tokenizer = getattr(runtime.processor, "tokenizer", None)
    if tokenizer is not None:
        try:
            return len(tokenizer.encode(text))
        except Exception:
            pass
    rough = len(text.strip().split())
    return rough if rough > 0 else 1


def _resolve_strip_thinking(request_value: bool | None) -> bool:
    if isinstance(request_value, bool):
        return request_value
    return STRIP_THINKING_DEFAULT


def _strip_thinking_content(text: str) -> str:
    if not text:
        return text

    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    marker_patterns = [
        r"(?is)^\s*thinking\s*process\s*:\s*",
        r"(?is)^\s*reasoning\s*:\s*",
        r"(?is)^\s*chain\s*of\s*thought\s*:\s*",
    ]
    for pattern in marker_patterns:
        cleaned = re.sub(pattern, "", cleaned, count=1).strip()

    return cleaned


def _safe_generate(
    messages: list[dict[str, Any]],
    prompt: str,
    images: list[Image.Image],
    videos: list[str],
    audios: list[str],
    max_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
) -> tuple[str, int | None, int | None]:
    kwargs: dict[str, Any] = {
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "verbose": False,
    }

    def generate_with_qwen3_omni_inputs() -> tuple[str, int | None, int | None]:
        from mlx_vlm.models.qwen3_omni_moe.omni_utils import prepare_omni_inputs

        conversation: list[dict[str, Any]] = []
        media_attached = False
        target_index = -1
        for index in range(len(messages) - 1, -1, -1):
            if messages[index].get("role") == "user":
                target_index = index
                break
        if target_index == -1 and messages:
            target_index = len(messages) - 1

        for index, message in enumerate(messages):
            role = str(message.get("role", "user"))
            content = message.get("content", "")
            text = content if isinstance(content, str) else ""

            if index == target_index and not media_attached and (audios or images or videos):
                parts: list[dict[str, Any]] = []
                for audio in audios:
                    parts.append({"type": "audio", "audio": audio})
                for image in images:
                    parts.append({"type": "image", "image": image})
                for video in videos:
                    parts.append({"type": "video", "video": video})
                if text:
                    parts.append({"type": "text", "text": text})
                conversation.append({"role": role, "content": parts if parts else text})
                media_attached = True
                continue

            conversation.append({"role": role, "content": text})

        model_inputs, _ = prepare_omni_inputs(runtime.processor, conversation)
        input_ids = model_inputs["input_ids"]
        mask = model_inputs.get("attention_mask")
        extra_kwargs = {
            key: value
            for key, value in model_inputs.items()
            if key not in {"input_ids", "attention_mask"}
        }

        thinker_result, _ = runtime.model.generate(
            input_ids=input_ids,
            mask=mask,
            return_audio=False,
            thinker_max_new_tokens=max_tokens,
            thinker_temperature=temperature,
            thinker_top_p=top_p,
            thinker_repetition_penalty=repetition_penalty,
            **extra_kwargs,
        )

        sequence = thinker_result.sequences[0].tolist()
        prompt_tokens = int(input_ids.shape[-1])
        generated_ids = sequence[prompt_tokens:]
        completion_tokens = len(generated_ids)

        processor = runtime.processor
        if hasattr(processor, "decode"):
            text = processor.decode(generated_ids)
        else:
            tokenizer = getattr(processor, "tokenizer", processor)
            text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        return text.strip(), prompt_tokens, completion_tokens

    try:
        if getattr(runtime.model.config, "model_type", "") == "qwen3_omni_moe" and (images or videos or audios):
            return generate_with_qwen3_omni_inputs()
        if videos and audios:
            raise ValueError("audio and video cannot be used together in the same request")
        if videos:
            processor = runtime.processor
            model_inputs = processor(
                text=[prompt],
                images=images if images else None,
                videos=videos,
                padding=True,
                return_tensors="pt",
            )

            input_ids = mx.array(model_inputs["input_ids"])
            pixel_values = model_inputs.get("pixel_values_videos", model_inputs.get("pixel_values", None))
            if pixel_values is not None:
                pixel_values = mx.array(pixel_values)
            attention_mask = model_inputs.get("attention_mask", None)
            mask = mx.array(attention_mask) if attention_mask is not None else None

            extra_kwargs: dict[str, Any] = {
                "input_ids": input_ids,
                "pixel_values": pixel_values,
                "mask": mask,
                "video": videos,
            }
            if model_inputs.get("video_grid_thw", None) is not None:
                extra_kwargs["video_grid_thw"] = mx.array(model_inputs["video_grid_thw"])
            if model_inputs.get("image_grid_thw", None) is not None:
                extra_kwargs["image_grid_thw"] = mx.array(model_inputs["image_grid_thw"])

            output = generate(runtime.model, runtime.processor, prompt, **kwargs, **extra_kwargs)
        elif audios:
            output = generate(runtime.model, runtime.processor, prompt, images if images else None, audio=audios, **kwargs)
        elif images:
            output = generate(runtime.model, runtime.processor, prompt, images, **kwargs)
        else:
            output = generate(runtime.model, runtime.processor, prompt, **kwargs)
    except TypeError:
        kwargs.pop("repetition_penalty", None)
        if getattr(runtime.model.config, "model_type", "") == "qwen3_omni_moe" and (images or videos or audios):
            return generate_with_qwen3_omni_inputs()
        if videos and audios:
            raise ValueError("audio and video cannot be used together in the same request")
        if videos:
            processor = runtime.processor
            model_inputs = processor(
                text=[prompt],
                images=images if images else None,
                videos=videos,
                padding=True,
                return_tensors="pt",
            )
            input_ids = mx.array(model_inputs["input_ids"])
            pixel_values = model_inputs.get("pixel_values_videos", model_inputs.get("pixel_values", None))
            if pixel_values is not None:
                pixel_values = mx.array(pixel_values)
            attention_mask = model_inputs.get("attention_mask", None)
            mask = mx.array(attention_mask) if attention_mask is not None else None
            extra_kwargs = {
                "input_ids": input_ids,
                "pixel_values": pixel_values,
                "mask": mask,
                "video": videos,
            }
            if model_inputs.get("video_grid_thw", None) is not None:
                extra_kwargs["video_grid_thw"] = mx.array(model_inputs["video_grid_thw"])
            if model_inputs.get("image_grid_thw", None) is not None:
                extra_kwargs["image_grid_thw"] = mx.array(model_inputs["image_grid_thw"])
            output = generate(runtime.model, runtime.processor, prompt, **kwargs, **extra_kwargs)
        elif audios:
            output = generate(runtime.model, runtime.processor, prompt, images if images else None, audio=audios, **kwargs)
        elif images:
            output = generate(runtime.model, runtime.processor, prompt, images, **kwargs)
        else:
            output = generate(runtime.model, runtime.processor, prompt, **kwargs)

    if isinstance(output, str):
        return output, None, None
    if isinstance(output, dict):
        if isinstance(output.get("text"), str):
            return output["text"], None, None
        return str(output), None, None

    generated_text = getattr(output, "text", None)
    prompt_tokens = getattr(output, "prompt_tokens", None)
    completion_tokens = getattr(output, "generation_tokens", None)

    if isinstance(generated_text, str):
        prompt_tokens = int(prompt_tokens) if isinstance(prompt_tokens, int | float) else None
        completion_tokens = int(completion_tokens) if isinstance(completion_tokens, int | float) else None
        return generated_text, prompt_tokens, completion_tokens

    return str(output), None, None


def _run_generation(
    messages: list[dict[str, Any]],
    images: list[Image.Image],
    videos: list[str],
    audios: list[str],
    max_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    strip_thinking: bool = False,
) -> tuple[str, int, int]:
    prompt = _safe_apply_chat_template(
        messages,
        len(images),
        len(videos),
        len(audios),
        disable_thinking=strip_thinking,
        video_source=videos[0] if videos else None,
    )
    text, prompt_tokens_raw, completion_tokens_raw = _safe_generate(
        messages=messages,
        prompt=prompt,
        images=images,
        videos=videos,
        audios=audios,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )
    prompt_tokens = prompt_tokens_raw if isinstance(prompt_tokens_raw, int) else _estimate_tokens(prompt)
    completion_tokens = completion_tokens_raw if isinstance(completion_tokens_raw, int) else _estimate_tokens(text)
    return text, prompt_tokens, completion_tokens


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/models")
def models_openai() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "owned_by": "local",
            }
        ],
    }


@app.get("/api/v1/models")
def models_lm_studio() -> dict[str, Any]:
    return {
        "models": [
            {
                "type": "llm",
                "key": MODEL_ID,
                "display_name": MODEL_ID,
                "max_context_length": MAX_CONTEXT_LENGTH,
                "capabilities": {"vision": True},
            }
        ]
    }


@app.post("/generate")
def generate_simple(request: GenerateRequest) -> dict[str, Any]:
    try:
        strip_thinking = _resolve_strip_thinking(request.strip_thinking)
        messages = [{"role": "user", "content": request.prompt}]
        text, prompt_tokens, completion_tokens = _run_generation(
            messages=messages,
            images=[],
            videos=[],
            audios=[],
            max_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
            strip_thinking=strip_thinking,
        )
        if strip_thinking:
            text = _strip_thinking_content(text)
        return {
            "text": text,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": {"message": f"generation failed: {exc}"}})


@app.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionsRequest) -> dict[str, Any]:
    _validate_model_id(request.model)
    if not request.messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")

    try:
        strip_thinking_request = request.strip_thinking
        if strip_thinking_request is None and isinstance(request.extra_body, dict):
            extra_strip = request.extra_body.get("strip_thinking")
            if isinstance(extra_strip, bool):
                strip_thinking_request = extra_strip
        strip_thinking = _resolve_strip_thinking(strip_thinking_request)

        messages, images, videos, audios = _extract_openai_input(request.messages)
        text, prompt_tokens, completion_tokens = _run_generation(
            messages=messages,
            images=images,
            videos=videos,
            audios=audios,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
            strip_thinking=strip_thinking,
        )
        if strip_thinking:
            text = _strip_thinking_content(text)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": {"message": str(exc)}})
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": {"message": f"chat generation failed: {exc}"}})

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


@app.post("/api/v1/chat")
def chat_lm_studio(request: LmStudioChatRequest) -> dict[str, Any]:
    _validate_model_id(request.model)

    try:
        strip_thinking = _resolve_strip_thinking(request.strip_thinking)
        content_parts, images, videos, audios = _extract_lm_studio_input(request.input)
        if not content_parts and not images and not videos and not audios:
            raise HTTPException(status_code=400, detail={"error": {"message": "input must include text message, image, audio, or video"}})

        messages: list[dict[str, Any]] = []
        if isinstance(request.system_prompt, str) and request.system_prompt.strip():
            messages.append({"role": "system", "content": request.system_prompt.strip()})

        messages.append({"role": "user", "content": _extract_text(content_parts)})

        text, prompt_tokens, completion_tokens = _run_generation(
            messages=messages,
            images=images,
            videos=videos,
            audios=audios,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
            strip_thinking=strip_thinking,
        )
        if strip_thinking:
            text = _strip_thinking_content(text)

        return {
            "output": [{"type": "message", "content": text}],
            "stats": {
                "input_tokens": prompt_tokens,
                "total_output_tokens": completion_tokens,
            },
        }
    except HTTPException:
        raise
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": {"message": str(exc)}})
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": {"message": f"chat generation failed: {exc}"}})
