# Create a virtural environment with PyTorch
```bash
conda create --name fite7001 -c conda-forge python=3.11
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

# Activate the virtual environment and install the dependencies
```bash
pip install jupyter
pip install --upgrade huggingface_hub
pip install langchainhub langchain sentence-transformers tiktoken chromadb GitPython langchain_experimental google-search-results
export CUDACXX="/usr/local/cuda/bin/nvcc"
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python
```