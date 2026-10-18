# Dashboard backend. Build from the repo root, because it also needs simulation/ and classifier/model:
#   docker build -f backend/Dockerfile -t robotics-backend .
# The datasets are not baked in: mount models/ at /app/models (docker-compose.yml does).

# Wheels are built in a separate stage so a package without a wheel for this platform (e.g. linux/arm64
# on Apple silicon) can still compile, without leaving the compiler in the final image.
FROM python:3.11-slim AS wheels
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
COPY backend/requirements.txt /tmp/requirements.txt
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r /tmp/requirements.txt

FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
# libgomp1: OpenMP runtime for xgboost and scikit-learn.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY --from=wheels /wheels /wheels
COPY backend/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --no-index --find-links /wheels -r /tmp/requirements.txt && rm -rf /wheels

# Same layout as the repo: backend/src/config.py finds everything relative to the repo root (/app).
WORKDIR /app
COPY simulation/hand_sim simulation/hand_sim
COPY simulation/Inspire_hand simulation/Inspire_hand
COPY simulation/objects simulation/objects
COPY classifier/model classifier/model
COPY backend backend

# 8000: HTTP + WebSocket. 5005/udp: the hand GUI's live broadcast (hardware/config.py).
EXPOSE 8000 5005/udp
HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4)"
CMD ["uvicorn", "backend.src.main:app", "--host", "0.0.0.0", "--port", "8000"]
