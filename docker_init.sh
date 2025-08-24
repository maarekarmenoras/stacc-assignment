#!/bin/sh
uv run load_data.py
uv run create_model.py
uv run main.py --port 5454 --host 0.0.0.0
