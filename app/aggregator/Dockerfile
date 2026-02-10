FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir kafka-python
COPY aggregator.py .
CMD ["python", "aggregator.py"]
