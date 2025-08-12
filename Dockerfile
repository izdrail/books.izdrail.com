FROM ubuntu:22.04

# Set working directory
WORKDIR /app

ENV API_URL=http://localhost:1602

# Install all system dependencies in a single layer
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    curl \
    ffmpeg \
    gcc \
    g++ \
    gnupg \
    make \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    python3-wheel \
    espeak-ng \
    libsndfile1-dev \
    supervisor && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    python3 --version

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gradio moviepy

# Pre-download the XTTS model during build
RUN python3 -c "import os; os.environ['COQUI_TOS_AGREED'] = '1'; from TTS.api import TTS; import torch; print('Pre-downloading XTTS model...'); tts = TTS('tts_models/multilingual/multi-dataset/xtts_v2'); print('XTTS model downloaded successfully!')"

# Copy Supervisor config
COPY docker/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Expose the port
EXPOSE 1602

# Copy application code
COPY . .

# Run Supervisor as the entry point
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]