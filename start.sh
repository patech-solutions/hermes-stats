#!/bin/bash
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"
exec /usr/bin/python3 -m uvicorn app:app --host 0.0.0.0 --port 8088 --reload
