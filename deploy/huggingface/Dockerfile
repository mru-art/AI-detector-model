# Hugging Face Space (Docker) that serves the S.C.R.A.M API on port 7860.
# Built from the folder that deploy/publish_space.py uploads.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Spaces run containers as user 1000.
RUN useradd -m -u 1000 user
USER user
WORKDIR /home/user/app
COPY --chown=user . .

ENV MODEL_PATH=vision_transformer/continual_learning/results/replay/final_model.pth \
    TORCH_THREADS=2 \
    MAX_TILES=9 \
    CORS_ORIGINS=*

EXPOSE 7860
CMD ["uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "7860"]
