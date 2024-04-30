# FITE7001 - Finance GPT to Automate Trading Strategy as Codes

## Description
This project aims to bridge that gap by introducing an accessible approach: utilizing LLM to automate trading strategies with no prior coding experience required, and with the goal of enhancing efficiency and accessibility. 

## Table of Contents

- [Installation](#installation)
- [Further Installation for Application](#further-installation-for-running-streamlit-application-not-necessary-for-running-playbooks)
- [Usage](#usage)
- [Running the Streamlit Application Locally](#running-the-streamlit-Application-locally)
- [Playbook Testing Result](#testing-result-of-the-playbooks)
- [License](#license)

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
---
## Further Installation For Running Streamlit Application (not necessary for running playbooks)
3. Install other dependencies

The `requirements.txt` file located in the `./Streamlit` folder contains the necessary Python packages required to run the Streamlit application. You can install these dependencies using the following command:

```bash
pip install -r Streamlit/requirements.txt
```
--- 

## Usage
**Important:** Please go to [TheBloke/Llama-2-13B-chat-GGUF](https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF) to download the `llama-2-13b-chat.Q5_K_M.gguf` model file from Huggingface before executing the playbooks.

After downloading the model file, place it in the `./model` directory of this project.

## Running the Streamlit Application locally

To run the application locally, follow these steps:

### Terminal 1

1. Open Terminal and SSH into the remote server:
    ```bash
    ssh -X [hku_account_id]@gpu2gate1.cs.hku.hk
    ```
    Enter password when prompted

2. Upload files to the remote server:
    ```bash
    scp -r {your_local_directory_location} {hku_account_id}@gpu2gate1.cs.hku.hk:{remote_directory_location}
    ```
    Upload your local project to remote server

3. Get the hostname of the server:
    ```bash
    hostname -i
    ```
    Note down the hostname (e.g., `10.xx.xx.xx`).


4. Log in to the GPU farm:
    ```bash
    srun --gres=gpu:2 --cpus-per-task=8 --pty --mail-type=ALL bash
    ```

5. Navigate to the project directory:
    ```bash
    cd ~/fite7001-project/finance-gpt/Streamlit
    ```

6. Activate the conda environment:
    ```bash
    conda activate fite7001
    ```

7. Run the Streamlit application:
    ```bash
    streamlit run StreamlitApp.py
    ```

### Terminal 2

1. Open a new Terminal window.

2. Set up port forwarding from the remote server to localhost:
    ```bash
    ssh -N -L 8501:localhost:8501 [hku_account_id]@10.xx.xx.xx
    ```
    Replace `10.xx.xx.xx` with the hostname obtained earlier.

### Browser

1. Open your web browser and go to:
    ```
    http://localhost:8501
    ```

This will launch the Streamlit application on your local machine, allowing you to interact with it via your web browser.

## Testing result of the playbooks
Testing results of the playbooks can be found in the /playbooks folder within the Jupyter notebooks.

## License
The Streamlit application in this project (`StreamlitApp.py`) includes code adapted from [Vikram Bhat's RAG Implementation with ConversationUI](https://github.com/vikrambhat2/RAG-Implementation-with-ConversationUI/blob/main/Streamlit%20Applications/StreamlitApp.py). Please refer to the original repository for more details on its licensing.

