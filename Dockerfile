# Use lightweight Python image
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Copy requirements first (layer caching optimization)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app

# Expose port
EXPOSE 8000

# Production start command
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
