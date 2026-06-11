#!/bin/bash
VENV=~/.hermes/hermes-agent/venv/bin/python
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"
exec "$VENV" -m uvicorn app:app --host 0.0.0.0 --port 8088 --reload
