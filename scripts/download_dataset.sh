#!/usr/bin/env bash
# Fetch DVBPR-formatted .npy files into a target folder.
# Usage:
#   bash scripts/download_dataset.sh --dataset all --out data
# Options:
#   --dataset {all|fashion|women|men|tradesy}  (default: all)
#   --out <DIR>                                (default: data)
# Notes:
#   - Files are from the original DVBPR links (UCSD).
#   - Creates the output folder if it does not exist.
#   - Requires curl or wget.

set -euo pipefail

DATASET="all"
OUTDIR="data"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2;;
    --out) OUTDIR="$2"; shift 2;;
    -h|--help)
      sed -n '1,80p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown arg: $1"
      exit 1
      ;;
  esac
done

mkdir -p "$OUTDIR"

have_cmd() { command -v "$1" >/dev/null 2>&1; }

fetch() {
  local url="$1"
  local out="$2"
  echo "-> Downloading $(basename "$out")"
  if have_cmd curl; then
    curl -L --fail --retry 3 -o "$out" "$url"
  elif have_cmd wget; then
    wget -nv -O "$out" "$url"
  else
    echo "Error: need curl or wget"
    exit 1
  fi
  echo "   Saved to $out"
}

FASHION_URL="http://cseweb.ucsd.edu/~wckang/DVBPR/AmazonFashion6ImgPartitioned.npy"
WOMEN_URL="http://cseweb.ucsd.edu/~wckang/DVBPR/AmazonWomenWithImgPartitioned.npy"
MEN_URL="http://cseweb.ucsd.edu/~wckang/DVBPR/AmazonMenWithImgPartitioned.npy"
TRADESY_URL="http://cseweb.ucsd.edu/~wckang/DVBPR/TradesyImgPartitioned.npy"

case "$DATASET" in
  all)
    fetch "$FASHION_URL" "$OUTDIR/AmazonFashion6ImgPartitioned.npy"
    fetch "$WOMEN_URL" "$OUTDIR/AmazonWomenWithImgPartitioned.npy"
    fetch "$MEN_URL" "$OUTDIR/AmazonMenWithImgPartitioned.npy"
    fetch "$TRADESY_URL" "$OUTDIR/TradesyImgPartitioned.npy"
    ;;
  fashion)
    fetch "$FASHION_URL" "$OUTDIR/AmazonFashion6ImgPartitioned.npy"
    ;;
  women)
    fetch "$WOMEN_URL" "$OUTDIR/AmazonWomenWithImgPartitioned.npy"
    ;;
  men)
    fetch "$MEN_URL" "$OUTDIR/AmazonMenWithImgPartitioned.npy"
    ;;
  tradesy)
    fetch "$TRADESY_URL" "$OUTDIR/TradesyImgPartitioned.npy"
    ;;
  *)
    echo "Unknown dataset: $DATASET"
    exit 1
    ;;
esac

echo "Done."
