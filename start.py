#!/usr/bin/env python3
"""
Small Python start wrapper to read PORT from environment and exec gunicorn with
an expanded numeric port. This avoids cases where the runtime passes a literal
"$PORT" into gunicorn (no shell expansion).

Usage: python start.py
Environment:
  PORT - optional, defaults to 5000
"""
import os
import sys
import shutil


def main():
    print("=== DEBUG: start.py is running ===")
    port = os.environ.get("PORT", "5000")
    print(f"=== DEBUG: Raw PORT from environment: '{port}' ===")
    # validate port is numeric-ish
    try:
        int(port)
        print(f"=== DEBUG: PORT '{port}' is valid integer ===")
    except Exception:
        print(f"=== DEBUG: Invalid PORT value '{port}', defaulting to 5000 ===")
        port = "5000"

    bind = f"0.0.0.0:{port}"
    print(f"=== DEBUG: Starting gunicorn on {bind} ===")
    print(f"=== DEBUG: Full environment PORT = {repr(os.environ.get('PORT'))} ===")

    # Ensure gunicorn is available on PATH
    gunicorn_path = shutil.which("gunicorn")
    if not gunicorn_path:
        print("gunicorn executable not found on PATH. Ensure it's installed in requirements.")
        sys.exit(2)

    # Replace the current process with gunicorn, preserving PID
    args = [
        gunicorn_path,
        "--workers",
        "4",
        "--bind",
        bind,
        "app:app",
    ]

    os.execv(gunicorn_path, args)


if __name__ == "__main__":
    main()
