FROM python:3.11-slim
WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY train.py ./
COPY app ./app
COPY monitoring ./monitoring
# Train at build time so the image is self-contained and versioned with its model.
# (Alternative: COPY a model from a registry/S3 instead of retraining.)
RUN python train.py --out artifacts
ENV ARTIFACT_DIR=/srv/artifacts PREDICTION_DB=/srv/data/predictions.db
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
