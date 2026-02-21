FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HUB_DISABLE_TELEMETRY=1 \
    PYTHONPYCACHEPREFIX=/dev/shm/pycache

# Minimal OS deps (opencv-python-headless often needs libglib; libgl can still be required in some cases)
RUN apt-get update && apt-get install -y --no-install-recommends \
      libglib2.0-0 \
      libgl1 \
      ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt /tmp/requirements.txt

# Upgrade pip and install deps
RUN python -m pip install --upgrade pip setuptools wheel \
 && python -m pip install --no-cache-dir -r /tmp/requirements.txt

# Optional: import smoke test (fails the build early if something is wrong)
RUN python -c "import diffusers, transformers, peft, accelerate, omegaconf, einops, numpy, PIL, tqdm, safetensors; import cv2; print('imports-ok')"

# Optional: precompile bytecode to speed up cold imports
RUN python - <<'PY'\n\
import importlib, pathlib, compileall\n\
mods = ['diffusers','transformers','peft','accelerate','omegaconf','einops','tqdm','safetensors','PIL']\n\
for m in mods:\n\
    mod = importlib.import_module(m)\n\
    p = pathlib.Path(mod.__file__).resolve()\n\
    root = p.parent if p.name != '__init__.py' else p.parent\n\
    compileall.compile_dir(str(root), quiet=1)\n\
print('compileall-ok')\n\
PY

# Do NOT override ENTRYPOINT/CMD so RunPod's base image behavior stays intact.