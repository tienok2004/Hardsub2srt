FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip ffmpeg libgl1 libgomp1 zlib1g locales \
    && locale-gen en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN python3 -m venv /opt/venv
COPY requirements.txt requirements-gpu.txt ./
RUN pip install -r requirements-gpu.txt

COPY core/ ./core/
COPY app.py cli.py config.yaml ./

# Build-time model warm-up downloads/validates the default detector and recognizer.
# CPU is used here because Docker build does not require a visible GPU.
RUN python -c "from PIL import Image; from rapidocr import RapidOCR; o=RapidOCR(); im=Image.new('RGB',(320,64),'white'); o(im,use_det=True,use_cls=False,use_rec=False); o(im,use_det=False,use_cls=False,use_rec=True)"

EXPOSE 7860
CMD ["python", "app.py"]

