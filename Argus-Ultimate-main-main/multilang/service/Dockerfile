# Argus multilang HTTP service - one container per language (LANGUAGE env)
FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir fastapi uvicorn pydantic
COPY protocol_logic.py .
COPY main.py .
ENV LANGUAGE=rust
ENV PORT=8011
EXPOSE 8011
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8011}"]
