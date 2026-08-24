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
elif [ "$DATASET" = "tinyshakespeare" ] || [ "$DATASET" = "shakespeare" ] || [ "$DATASET" = "tiny_shakespeare" ]; then
    URL="https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    TARGET_NAME="tinyshakespeare.txt"
    echo "⬇️  Downloading TinyShakespeare dataset to $DATA_DIR/$TARGET_NAME..."
    if [ -f "$TARGET_NAME" ]; then
        echo "✅ Already exists: $TARGET_NAME"
    else
        curl -fSL -o "$TARGET_NAME" "$URL"
    fi
    echo "🎉 TinyShakespeare dataset is ready at $DATA_DIR/$TARGET_NAME! ($(wc -c < "$TARGET_NAME" | tr -d ' ') bytes)"

elif [ "$DATASET" = "wikitext2" ] || [ "$DATASET" = "wikitext-2" ] || [ "$DATASET" = "wikitext" ]; then
    BASE_URL="https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2"
    FILES=("train.txt" "valid.txt" "test.txt")
    SUBDIR="$DATA_DIR/wikitext-2"
    mkdir -p "$SUBDIR"
    echo "⬇️  Downloading WikiText-2 dataset to $SUBDIR..."

    for FILE in "${FILES[@]}"; do
        TARGET_FILE="$SUBDIR/$FILE"
        if [ -f "$TARGET_FILE" ]; then
            echo "✅ Already exists: $TARGET_FILE"
        else
            echo "📥 Downloading $FILE..."
            curl -fSL -o "$TARGET_FILE" "$BASE_URL/$FILE"
        fi
    done
    echo "🎉 All WikiText-2 dataset files are ready in $SUBDIR!"

elif [ "$DATASET" = "tinystories" ] || [ "$DATASET" = "tiny_stories" ]; then
    URL="https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt"
    TARGET_NAME="tinystories_valid.txt"
    echo "⬇️  Downloading TinyStories (Validation slice ~19MB) to $DATA_DIR/$TARGET_NAME..."
    if [ -f "$TARGET_NAME" ]; then
        echo "✅ Already exists: $TARGET_NAME"
    else
        curl -fSL -o "$TARGET_NAME" "$URL"
    fi
    echo "🎉 TinyStories dataset is ready at $DATA_DIR/$TARGET_NAME! ($(wc -c < "$TARGET_NAME" | tr -d ' ') bytes)"

elif [ "$DATASET" = "alpaca" ] || [ "$DATASET" = "alpaca_data" ] || [ "$DATASET" = "alpaca_cleaned" ]; then
    URL="https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
    TARGET_NAME="alpaca_data.json"
    echo "⬇️  Downloading Stanford Alpaca SFT dataset to $DATA_DIR/$TARGET_NAME..."
    if [ -f "$TARGET_NAME" ]; then
        echo "✅ Already exists: $TARGET_NAME"
    else
        curl -fSL -o "$TARGET_NAME" "$URL"
    fi
    echo "🎉 Alpaca dataset is ready at $DATA_DIR/$TARGET_NAME! ($(wc -c < "$TARGET_NAME" | tr -d ' ') bytes)"

elif [ "$DATASET" = "llm" ] || [ "$DATASET" = "all_llm" ]; then
    echo "⬇️  Downloading all 4 LLM datasets (tinyshakespeare, wikitext-2, tinystories, alpaca)..."
    bash "${BASH_SOURCE[0]}" tinyshakespeare
    bash "${BASH_SOURCE[0]}" wikitext2
    bash "${BASH_SOURCE[0]}" tinystories
    bash "${BASH_SOURCE[0]}" alpaca
    echo "🎉 All LLM datasets downloaded successfully!"

else
    echo "❌ Unknown dataset: $DATASET"
    echo "Supported datasets:"
    echo "  - Vision: fashion_mnist, mnist"
    echo "  - LLM:    tinyshakespeare, wikitext2, tinystories, alpaca, all_llm"
    exit 1
fi
