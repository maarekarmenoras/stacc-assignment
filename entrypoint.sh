#!/bin/sh
uv run src/load_data.py
uv run src/create_model.py
exec uv run fastapi run src/main.py "$@"
