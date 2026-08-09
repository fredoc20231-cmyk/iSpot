FROM python:3.11-slim

# System dependencies for scientific Python + R
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev \
    libgeos-dev \
    libproj-dev \
    libgdal-dev \
    r-base \
    && rm -rf /var/lib/apt/lists/*

# R packages for BayesSpace and other R-based methods
RUN R -e "install.packages(c('BiocManager', 'Matrix', 'jsonlite', 'mclust'), repos='https://cloud.r-project.org')" && \
    R -e "BiocManager::install(c('BayesSpace', 'SpatialExperiment'), update=FALSE, ask=FALSE)"

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY ispot/ ./ispot/
COPY data/ ./data/

# Create job storage directory
RUN mkdir -p /app/ispot_jobs

# Expose port
EXPOSE 8100

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8100/api/health || exit 1

# Start server
CMD ["python", "-m", "uvicorn", "ispot.api:app", "--host", "0.0.0.0", "--port", "8100", "--log-level", "info"]
