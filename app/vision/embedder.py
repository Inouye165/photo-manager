"""CLIP embedding engine – dual-mode (scene + region).

Wraps HuggingFace ``transformers`` CLIP to produce normalised embeddings
for both whole images (scene) and detection crops (region).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, Optional, Sequence

import numpy as np
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

try:
    import torch
except ImportError:
    torch = None

try:
    from transformers import AutoProcessor, CLIPModel
except ImportError:
    AutoProcessor = None
    CLIPModel = None


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


class Embedder:
    """Lazy-loaded CLIP embedding engine."""

    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        dim: int = 768,
        device: Literal["auto", "cuda", "mps", "cpu"] = "auto",
    ) -> None:
        self.model_name = model_name
        self.expected_dim = dim
        self._device_pref = device
        self._model: Optional[object] = None
        self._processor: Optional[object] = None
        self._device: str = "cpu"
        self._half: bool = False

    @property
    def ready(self) -> bool:
        return self._model is not None and self._processor is not None

    # ── Device resolution ────────────────────────────────────────────────

    def _resolve_device(self) -> None:
        if torch is None:
            self._device, self._half = "cpu", False
            return
        pref = self._device_pref
        if pref in ("auto", "cuda") and torch.cuda.is_available():
            self._device, self._half = "cuda", True
        elif pref in ("auto", "mps") and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            self._device, self._half = "mps", False
        else:
            self._device, self._half = "cpu", False

    # ── Lazy load ────────────────────────────────────────────────────────

    def load(self) -> None:
        """Eagerly load model weights (called once at startup or first use)."""
        if self._model is not None:
            return
        if CLIPModel is None or AutoProcessor is None:
            raise RuntimeError("transformers not installed — cannot load CLIP")

        self._resolve_device()
        self._processor = AutoProcessor.from_pretrained(self.model_name)
        model = CLIPModel.from_pretrained(self.model_name)
        model.eval()
        if torch is not None:
            model = model.to(self._device)
            if self._device == "cuda" and self._half:
                model = model.half()
        self._model = model

    # ── Image embedding ──────────────────────────────────────────────────

    def embed_image(self, image: Image.Image | Path) -> np.ndarray:
        """Return a normalised CLIP image embedding (1-D float32 array)."""
        if isinstance(image, (str, Path)):
            with Image.open(image) as f:
                image = ImageOps.exif_transpose(f)
                if image.mode != "RGB":
                    image = image.convert("RGB")

        if CLIPModel is None or torch is None:
            return self._fallback_image(image)

        try:
            self.load()
            inputs = self._processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            with torch.inference_mode():
                if self._device == "cuda" and self._half:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        features = self._model.get_image_features(**inputs)
                else:
                    features = self._model.get_image_features(**inputs)

            vec = features.detach().float().cpu().numpy()[0]
            return _normalize(vec)
        except Exception as exc:
            logger.warning("Falling back to deterministic image embed: %s", exc)
            return self._fallback_image(image)

    # ── Text embedding ───────────────────────────────────────────────────

    def embed_text(self, query: str) -> np.ndarray:
        """Return a normalised CLIP text embedding."""
        query = (query or "").strip()
        if not query:
            raise ValueError("query must not be empty")

        if CLIPModel is None or torch is None:
            return self._fallback_text(query)

        try:
            self.load()
            inputs = self._processor(text=query, padding=True, return_tensors="pt")
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            with torch.inference_mode():
                if self._device == "cuda" and self._half:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        features = self._model.get_text_features(**inputs)
                else:
                    features = self._model.get_text_features(**inputs)

            vec = features.detach().float().cpu().numpy()[0]
            return _normalize(vec)
        except Exception as exc:
            logger.warning("Falling back to deterministic text embed: %s", exc)
            return self._fallback_text(query)

    # ── Fallbacks ────────────────────────────────────────────────────────

    @staticmethod
    def _fallback_image(image: Image.Image) -> np.ndarray:
        resized = image.resize((32, 32), Image.Resampling.BICUBIC)
        arr = np.asarray(resized, dtype=np.float32) / 255.0
        means = arr.mean(axis=(0, 1))
        stds = arr.std(axis=(0, 1))
        vec = np.concatenate((arr.reshape(-1), means, stds))
        return _normalize(vec)

    @staticmethod
    def _fallback_text(query: str) -> np.ndarray:
        dim = 32 * 32 * 3 + 6
        vec = np.zeros(dim, dtype=np.float32)
        for tok in query.lower().split():
            for i, b in enumerate(tok.encode("utf-8", errors="ignore")):
                vec[(b + i * 31) % dim] += b / 255.0
        if not vec.any():
            vec[0] = 1.0
        return _normalize(vec)
