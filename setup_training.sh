#!/usr/bin/env bash
set -euo pipefail

# Piper TTS Training Setup Script
# Usage: bash setup_training.sh [OPTIONS]
#
# This script sets up the Piper training environment, downloads a checkpoint,
# and preprocesses your dataset. After running, you'll be ready to train.

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Defaults
LANGUAGE="en"
DATASET_DIR="$PROJECT_DIR/dataset_en-GB"
TRAINING_DIR="$PROJECT_DIR/training"
PIPER_DIR="$PROJECT_DIR/piper"
CHECKPOINT_DIR="$PROJECT_DIR/checkpoints"
SAMPLE_RATE=22050
QUALITY="medium"
CHECKPOINT_URL="https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/en/en_GB/cori/high/cori-high-500.ckpt"
CHECKPOINT_FILE="en_GB-cori-high.ckpt"

usage() {
    cat <<EOF
Piper TTS Training Setup

Usage: bash $0 [OPTIONS]

Options:
  --language LANG        Language code (default: en)
  --dataset-dir DIR      Path to exported LJSpeech dataset (default: $DATASET_DIR)
  --training-dir DIR     Output directory for training files (default: $TRAINING_DIR)
  --piper-dir DIR        Where to clone Piper (default: $PIPER_DIR)
  --sample-rate RATE     Audio sample rate (default: 22050)
  --checkpoint-url URL   URL to download checkpoint from
  --skip-clone           Skip cloning Piper repo (if already done)
  --skip-checkpoint      Skip checkpoint download
  --skip-preprocess      Skip preprocessing
  -h, --help             Show this help

After setup, train via the web UI:
  python3 -m piper_recording_studio
  Visit http://localhost:8000/train and click 'Start Training'

Or train from the command line:
  cd $PIPER_DIR/src/python
  source .venv/bin/activate
  python3 -m piper_train \\
    --dataset-dir $TRAINING_DIR \\
    --accelerator gpu --devices 1 \\
    --batch-size 32 \\
    --validation-split 0.0 --num-test-examples 0 \\
    --max_epochs 1000 \\
    --resume_from_checkpoint $CHECKPOINT_DIR/$CHECKPOINT_FILE \\
    --checkpoint-epochs 10 --precision 32
EOF
    exit 0
}

SKIP_CLONE=false
SKIP_CHECKPOINT=false
SKIP_PREPROCESS=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --language) LANGUAGE="$2"; shift 2 ;;
        --dataset-dir) DATASET_DIR="$2"; shift 2 ;;
        --training-dir) TRAINING_DIR="$2"; shift 2 ;;
        --piper-dir) PIPER_DIR="$2"; shift 2 ;;
        --sample-rate) SAMPLE_RATE="$2"; shift 2 ;;
        --checkpoint-url) CHECKPOINT_URL="$2"; shift 2 ;;
        --skip-clone) SKIP_CLONE=true; shift ;;
        --skip-checkpoint) SKIP_CHECKPOINT=true; shift ;;
        --skip-preprocess) SKIP_PREPROCESS=true; shift ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

echo "========================================"
echo " Piper TTS Training Setup"
echo "========================================"
echo ""
echo "  Language:      $LANGUAGE"
echo "  Dataset:       $DATASET_DIR"
echo "  Training dir:  $TRAINING_DIR"
echo "  Piper dir:     $PIPER_DIR"
echo "  Sample rate:   $SAMPLE_RATE"
echo ""

# Check prerequisites
echo "[1/5] Checking prerequisites..."

if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 not found. Install Python 3.10+."
    exit 1
fi

MISSING_PKGS=""
if ! command -v ffmpeg &>/dev/null; then
    MISSING_PKGS="$MISSING_PKGS ffmpeg"
fi
if ! command -v espeak-ng &>/dev/null; then
    MISSING_PKGS="$MISSING_PKGS espeak-ng"
fi

if [ -n "$MISSING_PKGS" ]; then
    echo "  Missing system packages:$MISSING_PKGS"
    echo "  Attempting to install..."
    if sudo apt-get install -y $MISSING_PKGS; then
        echo "  System packages installed."
    else
        echo "ERROR: Could not install$MISSING_PKGS."
        echo "       Please run manually: sudo apt-get install$MISSING_PKGS"
        exit 1
    fi
fi

if ! command -v nvidia-smi &>/dev/null; then
    echo "WARNING: nvidia-smi not found. GPU training requires NVIDIA drivers + CUDA."
    echo "         You can still preprocess without a GPU."
fi

# Check dataset exists
if [ ! -f "$DATASET_DIR/metadata.csv" ]; then
    echo "ERROR: Dataset not found at $DATASET_DIR/metadata.csv"
    echo "       Run: python3 -m export_dataset --audio-glob '*.wav' output/en-GB/ $DATASET_DIR"
    exit 1
fi

DATASET_COUNT=$(wc -l < "$DATASET_DIR/metadata.csv")
echo "  Dataset: $DATASET_COUNT samples found"
echo ""

# Clone Piper
echo "[2/5] Setting up Piper..."

if [ "$SKIP_CLONE" = true ] && [ -d "$PIPER_DIR" ]; then
    echo "  Skipping clone (--skip-clone)"
elif [ -d "$PIPER_DIR/src/python" ]; then
    echo "  Piper already cloned at $PIPER_DIR"
else
    git clone https://github.com/rhasspy/piper.git "$PIPER_DIR"
fi

cd "$PIPER_DIR/src/python"

if [ ! -d ".venv" ]; then
    echo "  Creating virtual environment..."
    python3 -m venv .venv
fi

source .venv/bin/activate
# Piper requires pytorch-lightning~=1.7.0 which has invalid metadata
# that pip>=24.1 rejects. Pin pip to a compatible version.
pip3 install "pip>=23.0,<24.1" wheel setuptools -q
pip3 install -e . -q
# Pin compatible versions — Piper's deps pull in versions that are too new
pip3 install "numpy<2" "torchmetrics==0.11.4" six -q
# Install PyTorch with CUDA 12.8 for modern GPUs (RTX 40xx/50xx)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 -q

if [ -f "build_monotonic_align.sh" ]; then
    echo "  Building monotonic alignment..."
    bash build_monotonic_align.sh 2>/dev/null || echo "  (monotonic align build skipped or already built)"
fi

echo "  Piper training environment ready."
echo ""

# Download checkpoint
echo "[3/5] Downloading checkpoint..."

mkdir -p "$CHECKPOINT_DIR"

if [ "$SKIP_CHECKPOINT" = true ]; then
    echo "  Skipping checkpoint download (--skip-checkpoint)"
elif [ -f "$CHECKPOINT_DIR/$CHECKPOINT_FILE" ]; then
    echo "  Checkpoint already exists: $CHECKPOINT_DIR/$CHECKPOINT_FILE"
else
    echo "  Downloading: $CHECKPOINT_URL"
    echo "  This may take a few minutes (up to ~1 GB for high quality)..."
    wget -q --show-progress -O "$CHECKPOINT_DIR/$CHECKPOINT_FILE" "$CHECKPOINT_URL"
    echo "  Saved to: $CHECKPOINT_DIR/$CHECKPOINT_FILE"
fi
echo ""

# Preprocess
echo "[4/5] Preprocessing dataset..."

if [ "$SKIP_PREPROCESS" = true ]; then
    echo "  Skipping preprocessing (--skip-preprocess)"
elif [ -f "$TRAINING_DIR/config.json" ]; then
    echo "  Training directory already preprocessed: $TRAINING_DIR"
    echo "  Delete $TRAINING_DIR to re-preprocess."
else
    mkdir -p "$TRAINING_DIR"

    cd "$PIPER_DIR/src/python"
    source .venv/bin/activate

    python3 -m piper_train.preprocess \
        --language "$LANGUAGE" \
        --input-dir "$DATASET_DIR" \
        --output-dir "$TRAINING_DIR" \
        --dataset-format ljspeech \
        --single-speaker \
        --sample-rate "$SAMPLE_RATE"

    echo "  Preprocessing complete."
fi
echo ""

# Print training command
echo "[5/5] Setup complete!"
echo ""
echo "========================================"
echo " Ready to train!"
echo "========================================"
echo ""
echo "Option A: Start training from the web UI"
echo "  python3 -m piper_recording_studio"
echo "  Visit http://localhost:8000/train and click 'Start Training'"
echo ""
echo "Option B: Start training from the command line"
echo ""
echo "  cd $PIPER_DIR/src/python"
echo "  source .venv/bin/activate"
echo ""
echo "  python3 -m piper_train \\"
echo "    --dataset-dir $TRAINING_DIR \\"
echo "    --accelerator gpu \\"
echo "    --devices 1 \\"
echo "    --batch-size 32 \\"
echo "    --validation-split 0.0 \\"
echo "    --num-test-examples 0 \\"
echo "    --max_epochs 1000 \\"
echo "    --resume_from_checkpoint $CHECKPOINT_DIR/$CHECKPOINT_FILE \\"
echo "    --checkpoint-epochs 10 \\"
echo "    --precision 32"
echo ""
echo "Monitor training:"
echo "  tensorboard --logdir $TRAINING_DIR/lightning_logs"
echo ""
echo "After training, export to ONNX:"
echo "  python3 -m piper_train.export_onnx \\"
echo "    $TRAINING_DIR/lightning_logs/version_0/checkpoints/BEST_CHECKPOINT.ckpt \\"
echo "    $PROJECT_DIR/my_voice.onnx"
echo ""
echo "  cp $TRAINING_DIR/config.json $PROJECT_DIR/my_voice.onnx.json"
echo ""
echo "Test your voice:"
echo "  echo 'Hello world!' | piper -m $PROJECT_DIR/my_voice.onnx --output_file test.wav"
echo ""
