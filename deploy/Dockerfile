# containerization/Dockerfile
FROM python:3.9-slim

# Install Java for Spark
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    openjdk-11-jdk \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

# Create working directory
WORKDIR /app

# Install Python dependencies
COPY containerization/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set entry point
ENTRYPOINT ["python", "main.py"]