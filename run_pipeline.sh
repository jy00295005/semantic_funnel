#!/bin/bash

set -euo pipefail

if [ $# -lt 2 ]; then
  echo "Usage: ./run_pipeline.sh <dataset_stem> <domain_label>"
  echo "Example: ./run_pipeline.sh ai_input AI"
  exit 1
fi

DATASET_STEM="$1"
DOMAIN_LABEL="$2"
INPUT_FILE="data/${DATASET_STEM}.csv"

if [ ! -f "$INPUT_FILE" ]; then
  echo "Input file not found: $INPUT_FILE"
  exit 1
fi

python round1_round2_extract.py --domain "$DOMAIN_LABEL" --input "$INPUT_FILE"
python round3_cluster.py --domain "$DATASET_STEM"
python round4_canonicalize.py --domain "$DATASET_STEM"

echo "Semantic Funnel pipeline completed."
