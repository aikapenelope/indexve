# ManualIQ API Dockerfile
#
# Addresses issues:
# - 2.2: Pre-download Docling models (~1.1GB) at build time to prevent
#         runtime OOM crashes when models are downloaded on-demand.
# - 4.4: Pinned base image tag (no :latest) for supply chain security.

FROM python:3.12.8-slim AS base

# System dependencies for Docling and PDF processing.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpoppler-cpp-dev \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer caching).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download Docling AI models at build time (issue 2.2).
# This prevents the ~1.1GB download from happening at runtime,
# which can cause OOM crashes in memory-limited environments.
RUN python -c "\
from docling.document_converter import DocumentConverter; \
DocumentConverter(); \
print('Docling models pre-downloaded successfully')"

# Copy application code.
COPY manualiq/ manualiq/

# Health check endpoint.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()"

EXPOSE 8000

CMD ["uvicorn", "manualiq.main:app", "--host", "0.0.0.0", "--port", "8000"]
