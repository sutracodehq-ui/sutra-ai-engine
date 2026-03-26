#!/bin/bash
# scripts/download_bitnet_model.sh
# Downloads the 1.58-bit quantized model from HuggingFace

set -e

MODEL_DIR="./models"
MODEL_FILE="BitNet-b1.58-2B-4T.gguf"
MODEL_URL="https://huggingface.co/1bitLLM/bitnet_b1_58-3B/resolve/main/bitnet_b1_58-3B-Q8_0.gguf" # Example URL

mkdir -p "$MODEL_DIR"

if [ -f "$MODEL_DIR/$MODEL_FILE" ]; then
    echo "✅ BitNet model already exists at $MODEL_DIR/$MODEL_FILE"
    exit 0
fi

echo "🚀 Downloading BitNet 1.58-bit model..."
curl -L "$MODEL_URL" -o "$MODEL_DIR/$MODEL_FILE"

echo "✨ Model download complete!"
