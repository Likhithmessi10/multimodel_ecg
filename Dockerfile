# Use official PyTorch runtime as a parent image
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Set working directory in container
WORKDIR /app

# Install system dependencies for audio/signal processing
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy python dependencies list
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy current directory contents into the container
COPY . .

# Expose port for Streamlit clinical dashboard UI
EXPOSE 8501

# Command to execute training by default or launch streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
