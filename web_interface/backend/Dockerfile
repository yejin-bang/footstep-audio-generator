FROM python:3.10

WORKDIR /app

# Install system dependencies + git for CLIP
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    git \
    libsndfile1 \
    libsndfile1-dev \
    ffmpeg \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY ../../requirements.txt .

# Install packages with compatible versions
RUN pip install --no-cache-dir \
    fastapi==0.116.0 \
    uvicorn==0.35.0 \
    celery==5.5.3 \
    redis==7.1.0 \
    python-multipart==0.0.20 \
    aiofiles==24.1.0 \
    numpy==1.26.4 \
    torch==2.2.2 \
    torchvision==0.17.2 \
    mediapipe==0.10.21 \
    opencv-python==4.10.0.84 \
    scipy \
    soundfile \
    librosa \
    pydub \
    requests \
    transformers==4.27.4 \
    Pillow \
    ftfy \
    regex \
    tqdm

# Install CLIP from GitHub (as shown in your requirements.txt)
RUN pip install --no-cache-dir git+https://github.com/openai/CLIP.git

# Copy the entire project
COPY ../.. /project

# Set working directory to backend
WORKDIR /project/web_interface/backend

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]