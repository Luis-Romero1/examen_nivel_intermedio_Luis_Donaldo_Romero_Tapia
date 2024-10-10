FROM python:3.12.4-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN apt-get update && apt-get install -y curl

RUN pip install --no-cache-dir -r /app/requirements.txt

COPY config.yaml /app/config.yaml

COPY examen.py /app/examen.py
COPY api_service.py /app/api_service.py

EXPOSE 8000

CMD ["bash"]
