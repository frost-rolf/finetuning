# Use Huggingface Transformers PyTorch GPU base image
FROM huggingface/transformers-pytorch-gpu:latest

# Add repository for Python 3.10
RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update

# Install Python 3.10 and development tools
RUN apt-get install -y python3.10 python3.10-dev python3-pip

# Install JupyterLab and any additional dependencies using Python 3.10
RUN python3.10 -m pip install --upgrade pip && \
    python3.10 -m pip install jupyterlab

# Install dependencies from requirements.txt using Python 3.10
COPY requirements.txt /tmp/requirements.txt
RUN python3.10 -m pip install --no-cache-dir -r /tmp/requirements.txt

# Copy project files
COPY run_finetune_with_peft.ipynb /workspace/run_finetune_with_peft.ipynb
COPY data /workspace/data

# Set the working directory
WORKDIR /workspace

# Expose port 8888 for JupyterLab access
EXPOSE 8888

# Start JupyterLab using Python 3.10
CMD ["python3.10", "-m", "jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]

