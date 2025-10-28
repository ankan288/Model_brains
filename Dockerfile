FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements-pinned.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . /app

# Make start script executable
RUN chmod +x ./start.sh

EXPOSE 5000

# Use the Python start wrapper which reads PORT and execs gunicorn.
# Set ENTRYPOINT so platform start commands cannot override with an un-expanded $PORT.
ENTRYPOINT ["python", "start.py"]
