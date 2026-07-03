#!/bin/bash
# Multi-Modal ECG Diagnostics Setup and Start Script

echo "========================================================="
echo "Initializing PTB-XL ECG Late-Fusion Diagnostics Project"
echo "========================================================="

# 1. Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment 'venv'..."
    python -m venv venv
fi

# 2. Activate virtual environment
echo "Activating virtual environment..."
source venv/Scripts/activate || source venv/bin/activate

# 3. Install packages
echo "Installing python dependencies from requirements.txt..."
pip install -r requirements.txt

# 4. Download dataset (light mode: 1000 records)
echo "Downloading metadata and 1000 stratified ECG waveform records..."
python download_data.py --num_records 1000

# 5. Train all baseline models and proposed late-fusion network
echo "Training all baseline and late-fusion models..."
python trainer.py --epochs 15

# 6. Start the interactive Streamlit dashboard
echo "Launching local Streamlit diagnostic dashboard..."
streamlit run app.py
