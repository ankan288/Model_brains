#!/bin/sh
# Start script that expands PORT and launches gunicorn
# Default PORT to 5000 if not set
: "${PORT:=5000}"
echo "Starting server on port $PORT"
exec gunicorn --workers 4 --bind 0.0.0.0:$PORT app:app
