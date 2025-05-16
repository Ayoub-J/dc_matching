# Use a lightweight base Python image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Install system dependencies (if needed for PDF or DOCX parsing)
RUN apt-get update && apt-get install -y \
    build-essential \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -m nltk.downloader punkt

# Copy all project files
COPY . .

# Expose Streamlit default port
EXPOSE 8501

# Run your Streamlit app (adjust filename if needed)
CMD ["streamlit", "run", "stream_app.py"]
