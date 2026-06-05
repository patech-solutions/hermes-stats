#!/bin/bash
VENV=~/.hermes/hermes-agent/venv/bin/python
cd ~/.hermes/dashboard
exec "$VENV" -m uvicorn app:app --host 0.0.0.0 --port 8088 --reload
