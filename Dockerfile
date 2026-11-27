FROM ghcr.io/mlflow/mlflow:v2.9.2

WORKDIR /app

RUN pip install --no-cache-dir \
    "typer>=0.9.0" \
    "pydantic>=2.0.0" \
    "pydantic-settings>=2.0.0" \
    "pandera>=0.17.0" \
    "fastapi>=0.104.0" \
    "uvicorn>=0.24.0" \
    "sqlmodel>=0.0.12" \
    "httpx>=0.25.0" \
    "rich>=13.0.0" \
    "psutil>=5.9.0" && \
    echo "/app/src" > /usr/local/lib/python3.10/site-packages/_predgate.pth && \
    printf '#!/usr/bin/env python\nfrom predgate.cli import app\napp()\n' > /usr/local/bin/predgate && \
    chmod +x /usr/local/bin/predgate

COPY . .

# Generate demo data inside image (persisted via named volume predgate-data)
RUN python examples/credit-risk-demo/train.py

EXPOSE 8000

CMD ["uvicorn", "predgate.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
