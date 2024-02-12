# FITE7001 - Finance GPT to Automate Trading Strategy as Codes

This project aims to bridge that gap by introducing an accessible approach: utilizing LLM to automate trading strategies with no prior coding experience required, and with the goal of enhancing efficiency and accessibility.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Folder Structure](#folder-structure)




## Installation

    1. create a conda environment for the project and install pytorch with nvidia cuda toolkit version 12.1

```bash
conda create --name fite7001 -c conda-forge python=3.11
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

    2. Activate the virtual environment and install the dependencies (i.e., langchain, and hugginface etc.)

```bash
pip install jupyter
pip install --upgrade huggingface_hub
pip install langchainhub langchain sentence-transformers tiktoken chromadb GitPython langchain_experimental google-search-results
export CUDACXX="/usr/local/cuda/bin/nvcc"
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python
```
    
## Usage

The code playbook demonstrating how to run llama 2 locally is in `playbooks/llama2.ipynb`. Please follow the steps in the playbook to generate the trading strategies accordingly.

Important: Please go to https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF to download the `llama-2-13b-chat.Q5_K_M.gguf` from Huggingface before executing the playbooks.




## Folder Structure

1. `models` is where the model artifacts are stored. For example, LLM and embeddings are stored in this folder.
2. `code` is where the code snippets for the trading strategies located. For example, `vectorbt-test.py` is a simple moving average strategy. We will utilize the parser from langchain to feed these code into our prompt templates for LLM to have a better context. In the future, any code related to trading strategies are stored here.
3. `playbooks` contains mainly the jupyter notebook to showcase how the LLM works.
