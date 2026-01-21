FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Optimize build with requirements first
COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

# Copy source code
COPY src/ ./src/
COPY entrypoint/ ./entrypoint/
COPY config/ ./config/
COPY env.yaml .

# Copy model checkpoint if it exists (or ensure it's volume mounted)
COPY model_checkpoint.pth . 

EXPOSE 8000

# Run using the modularized entrypoint
CMD ["python", "-m", "entrypoint.inference"]
