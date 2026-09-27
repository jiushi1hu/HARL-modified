#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
python -m examples.train --algo happo --env cascade_reservoir "$@"
