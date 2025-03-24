#!/usr/bin/bash

set -e

command_exists() {
    command -v "$1" &> /dev/null
}

# Check if uv is installed first
if command_exists uv; then
    echo "Setting up environment using uv..."
    uv init --bare --python 3.10 .
    uv add --python 3.10 -r requirements.txt
    echo "Environment setup with uv completed successfully!"
    exit 0
fi

# If uv is not available, check for Python 3.10
echo "uv not found, falling back to venv..."

# Function to check if Python 3.10.* is available
check_python_version() {
    local python_cmd=$1
    if ! command_exists "$python_cmd"; then
        return 1
    fi
    
    local version
    version=$("$python_cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    if [[ "$version" == "3.10" ]]; then
        return 0
    else
        return 1
    fi
}


# Find Python 3.10
PYTHON_CMD=""
echo "Searching for Python 3.10..."
for cmd in python3.10 python3 python; do
    echo "Trying command: $cmd"
    if check_python_version "$cmd"; then
        PYTHON_CMD="$cmd"
        echo "Found compatible Python with $cmd"
        break
    else
        echo "  $cmd is not available or not version 3.10.x"
    fi
done

if [[ -z "$PYTHON_CMD" ]]; then
    echo "Error: Python 3.10.* is required but not found on your system."
    exit 1
fi

echo "Using Python: $($PYTHON_CMD --version)"

# Create a virtual environment using venv
"$PYTHON_CMD" -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

echo "Environment setup with venv completed successfully!"
echo "Activate with: source .venv/bin/activate"
