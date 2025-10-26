FROM quay.io/astronomer/astro-runtime:11.20.0-ubi9-python-3.11-slim
# Copy requirements and install additional packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
