# Official RunPod CUDA PyTorch base (linux/amd64). Torch/CUDA come from the image —
# do not reinstall torch from PyPI (risk of CPU wheel overwrite).
FROM --platform=linux/amd64 runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /

COPY requirements.txt /requirements.txt
COPY handler.py /handler.py

# Install embedding stack only. Base image already provides CUDA torch.
# --extra-index-url keeps any transitive torch resolution on CUDA wheels if needed.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /requirements.txt \
        --extra-index-url https://download.pytorch.org/whl/cu124 \
    && python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

CMD ["python", "-u", "/handler.py"]
