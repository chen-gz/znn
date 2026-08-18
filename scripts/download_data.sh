#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:-fashion_mnist}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$ROOT_DIR/data"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

if [ "$DATASET" = "fashion_mnist" ] || [ "$DATASET" = "fashion-mnist" ]; then
    BASE_URL="http://fashion-mnist.s3-website.eu-central-1.amazonaws.com"
    FILES=(
        "train-images-idx3-ubyte.gz"
        "train-labels-idx1-ubyte.gz"
        "t10k-images-idx3-ubyte.gz"
        "t10k-labels-idx1-ubyte.gz"
    )

    echo "⬇️  Downloading Fashion MNIST dataset to $DATA_DIR..."

    for FILE in "${FILES[@]}"; do
        TARGET_NAME="${FILE%.gz}"
        if [ -f "$TARGET_NAME" ]; then
            echo "✅ Already exists: $TARGET_NAME"
        else
            echo "📥 Downloading $FILE..."
            curl -fSL -o "$FILE" "$BASE_URL/$FILE"
            echo "📦 Extracting $FILE..."
            gunzip -f "$FILE"
        fi
    done
    echo "🎉 All Fashion MNIST dataset files are ready in $DATA_DIR!"

elif [ "$DATASET" = "mnist" ]; then
    BASE_URL="https://storage.googleapis.com/cvdf-datasets/mnist"
    FILES=(
        "train-images-idx3-ubyte.gz"
        "train-labels-idx1-ubyte.gz"
        "t10k-images-idx3-ubyte.gz"
        "t10k-labels-idx1-ubyte.gz"
    )

    echo "⬇️  Downloading classic MNIST dataset to $DATA_DIR..."

    for FILE in "${FILES[@]}"; do
        TARGET_NAME="${FILE%.gz}"
        if [ -f "$TARGET_NAME" ]; then
            echo "✅ Already exists: $TARGET_NAME"
        else
            echo "📥 Downloading $FILE..."
            curl -fSL -o "$FILE" "$BASE_URL/$FILE"
            echo "📦 Extracting $FILE..."
            gunzip -f "$FILE"
        fi
    done
    echo "🎉 All MNIST dataset files are ready in $DATA_DIR!"
else
    echo "❌ Unknown dataset: $DATASET"
    echo "Supported datasets: fashion_mnist, mnist"
    exit 1
fi
