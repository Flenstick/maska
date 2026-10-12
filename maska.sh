#!/bin/sh
exec "$(dirname "$(readlink -f "$0")")/.venv/bin/python" "$(dirname "$(readlink -f "$0")")/maska.py" "$@"
