#!/bin/bash
"""Fin-JEPA HF Jobs submission script.

Usage:
    bash submit_hf.sh          # Submit full sweep to T4-small
    bash submit_hf.sh --dry    # Print command without running
"""
set -e

TAG="${1:-sweep1}"
VARIANTS="${2:-0}"   # 0 = all
HARDWARE="t4-small"

CMD="hf jobs uv run --resource $HARDWARE \
    --name fin-jepa-${TAG} \
    sweep.py --tag ${TAG} --variants ${VARIANTS}"

echo "=== Fin-JEPA HF Job ==="
echo "Hardware: $HARDWARE (\$0.40/hr)"
echo "Tag:      $TAG"
echo "Variants: ${VARIANTS:-all}"
echo "Command:  $CMD"
echo ""

if [ "$1" != "--dry" ]; then
    echo "Submitting..."
    cd /Users/hermes/dev/chan-jepa/hf
    eval $CMD
fi
