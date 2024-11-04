#!/bin/bash
#Usage: bash ./run.sh data/public.jsonl mt5_submission.jsonl

if [ "$#" -ne 2 ]; then
    echo "Usage: bash ./run.sh /path/to/input.jsonl /path/to/output.jsonl"
    exit 1
fi

INPUT_FILE="$1"
OUTPUT_FILE="$2"

python src/inf_mt5.py "$INPUT_FILE" "$OUTPUT_FILE"
