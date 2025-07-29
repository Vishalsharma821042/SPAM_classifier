# Use official Python base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Create app directory
WORKDIR /app

# Copy all project files
COPY . /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install NLTK dependencies
RUN python -m pip install --upgrade pip

# Install Python packages
RUN pip install -r requirements.txt

# Download NLTK data
RUN python -m nltk.downloader punkt stopwords

# Expose the port (Render uses a dynamic port via $PORT)
EXPOSE 8501

# Run Streamlit app with Render-compatible settings
CMD streamlit run spam.py --server.address=0.0.0.0 --server.port=$PORT
