# 23-Language HTTP Services (Docker)

The bot can call 23-language tasks over HTTP when `multi_language.endpoints` points to running services. By default it uses **in-process** Python handlers so the bot works without any external services.

## Reference service (one per language)

- **Code:** `multilang/service/main.py` (FastAPI), `multilang/service/protocol_logic.py`
- **Endpoints:** `GET /health`, `GET /ready`, `GET /metrics`, `GET /capabilities`, `POST /execute`, `POST /batch`, `POST /warm`
- **Run locally:**  
  `cd multilang/service && LANGUAGE=rust PORT=8011 uvicorn main:app --host 0.0.0.0 --port 8011`

## Docker (example)

Example compose file runs 3 services (rust, cpp, python) with the same Python logic; you can replace the image with real Rust/C++ implementations later.

```bash
docker-compose -f scripts/docker-compose-multilang.example.yml up -d
```

Then in `unified_config.yaml` under `multi_language.endpoints`:

```yaml
multi_language:
  endpoints:
    rust: "http://localhost:8011"
    cpp: "http://localhost:8012"
    python: "http://localhost:8013"
```

## Build (from repo root)

```bash
docker build -t argus-multilang:rust -f multilang/service/Dockerfile multilang/service
docker run -e LANGUAGE=rust -e PORT=8011 -p 8011:8011 argus-multilang:rust
```

See `scripts/docker-compose-multilang.example.yml` for a full example.
