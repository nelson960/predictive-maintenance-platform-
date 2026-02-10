FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY train_anomaly.py .
CMD ["python", "train_anomaly.py"]
