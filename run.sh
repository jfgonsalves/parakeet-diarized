#!/bin/bash
set -e

# Colors for terminal output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default settings
DEBUG=0
PORT=8000
HOST="0.0.0.0"
CHECK_DEPS=1
HF_TOKEN=""
ASR_BACKEND="parakeet"
MODEL_ID=""

# Process command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --debug)
            DEBUG=1
            shift
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --skip-deps-check)
            CHECK_DEPS=0
            shift
            ;;
        --hf-token)
            HF_TOKEN="$2"
            shift 2
            ;;
        --backend)
            ASR_BACKEND="$2"
            shift 2
            ;;
        --model)
            MODEL_ID="$2"
            shift 2
            ;;
        --help)
            echo -e "${BLUE}Parakeet Whisper-Compatible API Server${NC}"
            echo -e "Usage: $0 [options]"
            echo -e "Options:"
            echo -e "  --debug             Enable debug mode"
            echo -e "  --port PORT         Set server port (default: 8000)"
            echo -e "  --host HOST         Set server host (default: 0.0.0.0)"
            echo -e "  --backend BACKEND   ASR backend to use: parakeet or medasr (default: parakeet)"
            echo -e "  --model MODEL_ID    Model ID to use (default: nvidia/parakeet-tdt-0.6b-v2 or google/medasr)"
            echo -e "  --skip-deps-check   Skip dependency checking"
            echo -e "  --hf-token TOKEN    Set HuggingFace access token for speaker diarization"
            echo -e "  --help              Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $key${NC}" >&2
            exit 1
            ;;
    esac
done

echo -e "${GREEN}Starting Parakeet Whisper-Compatible API Server${NC}"

# Check for ffmpeg
if [[ $CHECK_DEPS -eq 1 ]]; then
    echo -e "${BLUE}Checking dependencies...${NC}"
    if ! command -v ffmpeg &> /dev/null; then
        echo -e "${RED}ERROR: ffmpeg is required but not installed.${NC}"
        echo -e "${YELLOW}Please install ffmpeg using your package manager:${NC}"
        echo -e "${YELLOW}  - Ubuntu/Debian: sudo apt-get install ffmpeg${NC}"
        echo -e "${YELLOW}  - MacOS: brew install ffmpeg${NC}"
        exit 1
    else
        echo -e "${GREEN}ffmpeg found: $(ffmpeg -version | head -n 1)${NC}"
    fi
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv || { echo -e "${RED}Failed to create virtual environment. Make sure python3-venv is installed.${NC}"; exit 1; }
fi

# Activate virtual environment
echo -e "${BLUE}Activating virtual environment...${NC}"
source venv/bin/activate || { echo -e "${RED}Failed to activate virtual environment.${NC}"; exit 1; }

# Install requirements if needed
if [ ! -f "venv/.requirements_installed" ] && [[ $CHECK_DEPS -eq 1 ]]; then
    echo -e "${YELLOW}Installing requirements...${NC}"
    pip install -r requirements.txt || { echo -e "${RED}Failed to install requirements.${NC}"; exit 1; }
    touch venv/.requirements_installed
fi

# Check for CUDA
if python -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
    echo -e "${GREEN}CUDA is available. Using GPU.${NC}"
    CUDA_INFO=$(python -c "import torch; print(torch.cuda.get_device_name(0))")
    echo -e "${GREEN}GPU Device: ${CUDA_INFO}${NC}"
else
    echo -e "${YELLOW}WARNING: CUDA is not available. Using CPU, which will be much slower.${NC}"
    echo -e "${YELLOW}Consider installing CUDA for better performance.${NC}"
fi

# Set environment variables
if [[ $DEBUG -eq 1 ]]; then
    echo -e "${YELLOW}Debug mode enabled. Verbose output will be shown.${NC}"
    export DEBUG=1
fi

# Set HuggingFace access token if provided
if [[ -n "$HF_TOKEN" ]]; then
    echo -e "${GREEN}HuggingFace access token set. Speaker diarization will be available.${NC}"
    export HUGGINGFACE_ACCESS_TOKEN="$HF_TOKEN"
fi

# Set ASR backend
export ASR_BACKEND="$ASR_BACKEND"
if [[ "$ASR_BACKEND" == "medasr" ]]; then
    echo -e "${GREEN}Using MedASR backend (medical speech recognition)${NC}"
else
    echo -e "${GREEN}Using Parakeet backend${NC}"
fi

# Set model ID if provided
if [[ -n "$MODEL_ID" ]]; then
    export MODEL_ID="$MODEL_ID"
    echo -e "${GREEN}Using model: $MODEL_ID${NC}"
fi

# Run the server
echo -e "${GREEN}Starting server on ${HOST}:${PORT}...${NC}"
if [[ $DEBUG -eq 1 ]]; then
    uvicorn main:app --host $HOST --port $PORT --reload --log-level debug
else
    uvicorn main:app --host $HOST --port $PORT --reload
fi
