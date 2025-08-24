#!/bin/sh
uv run load_data.py
uv run create_model.py
exec uv run fastapi run main.py "$@"
