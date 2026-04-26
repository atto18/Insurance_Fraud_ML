FROM python:3.11-slim

WORKDIR /app

# Install CPU-only torch first (separate cached layer — avoids the 2.5 GB CUDA wheel)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies (torch already satisfied, so pip skips it)
COPY requirements.txt .
RUN pip install --no-cache-dir --timeout 300 --retries 7 -r requirements.txt

# Copy source code, models, and pre-computed outputs
COPY src/ ./src/
COPY dashboard.py dashboard_core.py dashboard_pro.py ./
COPY models/ ./models/
COPY outputs/ ./outputs/

# Create data directories (user uploads raw data at runtime via the sidebar)
RUN mkdir -p data/raw data/preprocessed data/final

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "dashboard.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
