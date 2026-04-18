#!/bin/bash
echo "============================================"
echo " Simple Transcription - Setup (Linux/Mac)"
echo "============================================"
echo

# Step 1: Create virtual environment
echo "[1/5] Creating virtual environment..."
python3 -m venv venv || { echo "ERROR: Failed to create venv."; exit 1; }

# Step 2: Activate virtual environment
echo "[2/5] Activating virtual environment..."
source venv/bin/activate

# Step 3: Upgrade pip
echo "[3/5] Upgrading pip..."
python -m pip install --upgrade pip

# Step 4: Install OpenVINO first (pinned version)
echo "[4/5] Installing OpenVINO 2026.1.0..."
pip install openvino==2026.1.0

# Step 5: Install remaining dependencies
echo "[5/5] Installing remaining dependencies..."
pip install -r requirements.txt

echo
echo "============================================"
echo " Setup complete!"
echo
echo " To run the app:"
echo "   source venv/bin/activate"
echo "   python app.py"
echo
echo " (Optional) Pre-export models for faster first run:"
echo "   python scripts/export_models.py"
echo "============================================"
