# FROM python:3.9-slim

# # Install system dependencies for OpenCV
# RUN apt-get update && apt-get install -y \
#     libglib2.0-0 \
#     libsm6 \
#     libxext6 \
#     libxrender-dev \
#     libgomp1 \
#     libglib2.0-0 \
#     libgtk-3-0 \
#     && rm -rf /var/lib/apt/lists/*

# WORKDIR /app

# # Copy requirements first for better caching
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy application code
# COPY main.py main.py

# # Create outputs directory
# RUN mkdir -p outputs

# # Expose port
# EXPOSE 8000

# # Use optimized uvicorn settings for production
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--access-log", "--no-access-log"]



# Multi-stage build to reduce final image size
FROM python:3.9-slim as builder

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Final stage - minimal runtime image
FROM python:3.9-slim

# Install only runtime system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Set PATH to include local packages
ENV PATH=/root/.local/bin:$PATH

WORKDIR /app

# Copy application code
COPY main.py main.py

# Create outputs directory
RUN mkdir -p outputs

# Set environment variables for optimization
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Expose port
EXPOSE $PORT

# Use Railway's PORT environment variable
CMD uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1