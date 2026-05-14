# Single image for all 23 languages; LANGUAGE and PORT set per container.
FROM python:3.11-slim

WORKDIR /app
COPY service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY service/ .

ENV LANGUAGE=rust
ENV PORT=8011
EXPOSE 8011

CMD ["sh", "-c", "exec python -m uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
