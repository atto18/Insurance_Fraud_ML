FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

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
