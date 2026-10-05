# ==============================================================================
# Dockerfile: Medical Research Assistant (Agentic RAG System)
# Fully Compatible with Hugging Face Spaces & Production Containers
# ==============================================================================

FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable real-time unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_PORT=7860 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    HOME=/home/user

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set up a non-root user required for Hugging Face Spaces security (UID 1000)
RUN useradd -m -u 1000 user

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install pip-system-certs && \
    pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz

# Copy the rest of the application
COPY . /app

# Create data directories and grant full permissions to the non-root user
RUN mkdir -p /app/data/raw_pdfs \
             /app/data/incomplete_pdfs \
             /app/data/processed/faiss_index \
             /app/data/processed/chroma_db && \
    chown -R user:user /app /home/user && \
    chmod -R 777 /app/data

# Switch to non-root user
USER user

# Hugging Face Spaces listens on port 7860
EXPOSE 7860

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl --fail http://localhost:7860/_stcore/health || exit 1

# Launch Streamlit app
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=7860", "--server.address=0.0.0.0"]
