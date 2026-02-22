FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

LABEL org.opencontainers.image.source=https://github.com/mikaba-eu/hypir-torch-2-4-0
LABEL org.opencontainers.image.description="RunPod image for HYPIR batch upscaling"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HUB_DISABLE_TELEMETRY=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      libglib2.0-0 \
      libgl1 \
      ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt /tmp/requirements.txt

RUN python -m pip install --upgrade pip setuptools wheel \
 && python -m pip install --no-cache-dir -r /tmp/requirements.txt

RUN python -c "import diffusers, transformers, peft, accelerate, omegaconf, einops, numpy, PIL, tqdm, safetensors; import cv2; print('imports-ok')"

RUN env -u PYTHONPYCACHEPREFIX python - <<'PY'
import importlib
import pathlib
import compileall

mods = [
    "diffusers",
    "transformers",
    "peft",
    "accelerate",
    "omegaconf",
    "einops",
    "tqdm",
    "safetensors",
    "PIL",
]
for m in mods:
    mod = importlib.import_module(m)
    p = pathlib.Path(mod.__file__).resolve()
    compileall.compile_dir(str(p.parent), quiet=1)

print("compileall-ok")
PY