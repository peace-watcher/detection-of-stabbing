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

## 📚 Open Source Projects Used

Our Grounding DINO server utilizes the following open source projects:

1. [**Grounding DINO**](https://github.com/IDEA-Research/GroundingDINO.git):
    - Grounding DINO is an open-source project developed by IDEA-Research for object detection using a transformer-based approach.
    - We have cloned this repository and installed its requirements to integrate its capabilities into our server.
    - This project forms the core of our detection system, leveraging state-of-the-art techniques for accurate and efficient object detection.
2. [**FastAPI**](https://github.com/tiangolo/fastapi):
    - FastAPI is a modern, fast (high-performance), web framework for building APIs with Python 3.6+ based on standard Python type hints.
    - We use FastAPI to build the backend of our server, enabling us to create a robust and efficient API for our application.
3. [**Uvicorn**](https://github.com/encode/uvicorn):
    - Uvicorn is a lightning-fast ASGI server implementation, using `uvloop` and `httptools`.
    - It is used to run our FastAPI application, providing high performance and support for async programming.
4. [**Torch**](https://github.com/pytorch/pytorch):
    - Torch is an open-source machine learning library, primarily developed by Facebook's AI Research lab (FAIR).
    - We use Torch and its companion library, torchvision, for implementing deep learning models and handling computer vision tasks.
5. [**OpenCV**](https://github.com/opencv/opencv):
    - OpenCV (Open Source Computer Vision Library) is an open-source computer vision and machine learning software library.
    - OpenCV is utilized for various image processing tasks in our server.

These open-source projects are essential components of our Grounding DINO server, providing the necessary tools and frameworks to achieve our objectives.
