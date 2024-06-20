# Grounding DINO Python Server

## 🪧 About Source Code

### 👩‍💻 Prerequisites

- **Python Version**: 3.10.12
- **Server Specifications**: GPU v100, Ubuntu 20.04

### 🔧 How to Set Up

1. **Install Python 3.10.12 with pyenv**:
    
    ```bash
    curl https://pyenv.run | bash
    
    echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
    echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
    echo 'eval "$(pyenv init --path)"' >> ~/.bashrc
    echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc
    
    source ~/.bashrc
    
    pyenv install 3.10.12
    pyenv global 3.10.12
    ```
    
    ```plaintext
    # If you encounter a message to install gcc, follow the steps below.
    ```
    
2. **Install C Compiler**:
    
    ```bash
    sudo apt update
    sudo apt install build-essential
    
    gcc --version
    ```
    
3. **Install Required Libraries**:
    
    ```bash
    sudo apt install -y build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev wget libbz2-dev
    sudo apt install liblzma-dev
    
    sudo apt install make build-essential libssl-dev zlib1g-dev libbz2-dev \
    libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
    xz-utils tk-dev libffi-dev liblzma-dev python-openssl git
    ```
    
4. **Install Python**:
    
    ```bash
    pyenv install 3.10.12
    pyenv global 3.10.12
    ```
    
    ```bash
    python -c "import lzma; print('lzma module is installed')"
    ```
    
5. **Install Required Python Packages**:
    
    ```bash
    pip install fastapi uvicorn opencv-python-headless python-multipart
    pip install torch torchvision
    ```
    
6. **Clone the Repository**:
    
    ```bash
    git clone https://github.com/peace-watcher/detection-of-stabbing
    cd GroundingDINO
    pip install -r requirements.txt
    pip install -e .
    poetry install
    ```

7. **NVIDIA CUDA Installation**:
    Follow the instructions to download and install CUDA from the [NVIDIA CUDA Downloads page](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=deb_network).

### 🚀 How to Run

**CUDA Compatibility Check**:
Ensure CUDA 11.3 is installed and compatible with the required PyTorch version. Set up environment variables:

```bash
export PATH=/usr/local/cuda-11.3/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.3/lib64:$LD_LIBRARY_PATH
```

**Run the FastAPI Application**:

```bash
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```
