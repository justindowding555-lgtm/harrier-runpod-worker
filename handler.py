"""RunPod Serverless worker for microsoft/harrier-oss-v1-27b embeddings.

Requires a recent Transformers stack (gemma3_text / bidirectional attention):
transformers>=4.57.0 (prefer 5.x with sentence-transformers>=5.1).
"""

from __future__ import annotations

import os
from typing import Any

import runpod
import torch
from sentence_transformers import SentenceTransformer


def _configure_hf_cache() -> None:
    """Prefer RunPod network-volume / local HF caches when present."""
    candidates = [
        os.environ.get("HF_HOME"),
        os.environ.get("TRANSFORMERS_CACHE"),
        os.environ.get("HUGGINGFACE_HUB_CACHE"),
        "/runpod-volume/huggingface-cache",
        "/runpod-volume/hf-cache",
        "/workspace/huggingface-cache",
        "/models",
    ]
    for path in candidates:
        if path and os.path.isdir(path):
            os.environ.setdefault("HF_HOME", path)
            os.environ.setdefault("TRANSFORMERS_CACHE", path)
            os.environ.setdefault("HUGGINGFACE_HUB_CACHE", path)
            # Offline when the volume already has the model tree
            model_name = os.environ.get("MODEL_NAME", "microsoft/harrier-oss-v1-27b")
            local_snapshot = os.path.join(path, "hub", f"models--{model_name.replace('/', '--')}")
            if os.path.isdir(local_snapshot) or os.path.isdir(os.path.join(path, model_name)):
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
                os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            break


_configure_hf_cache()

MODEL_NAME = os.environ.get("MODEL_NAME", "microsoft/harrier-oss-v1-27b")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = SentenceTransformer(
    MODEL_NAME,
    model_kwargs={"dtype": "auto"},
    device=DEVICE,
)


def _normalize_texts(job_input: dict[str, Any]) -> list[str]:
    if "input" in job_input:
        raw = job_input["input"]
    elif "texts" in job_input:
        raw = job_input["texts"]
    else:
        raise ValueError(
            "Missing text input. Provide job['input']['input'] or job['input']['texts'] "
            "as a string or list of strings."
        )

    if isinstance(raw, str):
        texts = [raw]
    elif isinstance(raw, list):
        if not raw:
            raise ValueError("Text list is empty.")
        if not all(isinstance(t, str) for t in raw):
            raise ValueError("All items in input/texts must be strings.")
        texts = raw
    else:
        raise ValueError("input/texts must be a string or a list of strings.")

    return texts


def handler(job: dict[str, Any]) -> dict[str, Any]:
    job_input = job.get("input")
    if not isinstance(job_input, dict):
        return {"error": "job['input'] must be an object with 'input' or 'texts'."}

    try:
        texts = _normalize_texts(job_input)
    except ValueError as exc:
        return {"error": str(exc)}

    encode_kwargs: dict[str, Any] = {"convert_to_numpy": True}
    # Custom prompt string overrides named prompt (SentenceTransformer behavior).
    if job_input.get("prompt"):
        encode_kwargs["prompt"] = job_input["prompt"]
    elif job_input.get("prompt_name"):
        encode_kwargs["prompt_name"] = job_input["prompt_name"]

    embeddings = model.encode(texts, **encode_kwargs)
    if getattr(embeddings, "ndim", 1) == 1:
        embeddings = embeddings.reshape(1, -1)

    return {
        "embeddings": embeddings.tolist(),
        "dimensions": int(embeddings.shape[-1]),
        "model": MODEL_NAME,
    }


runpod.serverless.start({"handler": handler})
