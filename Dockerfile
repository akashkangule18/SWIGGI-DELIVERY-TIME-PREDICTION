# Base Image
FROM python:3.13-slim

# Install required Linux libraries
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Working Directory
WORKDIR /app

# Copy project
COPY . .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Expose FastAPI port
EXPOSE 8000

# Start API
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]