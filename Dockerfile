# syntax=docker/dockerfile:1

# Start from the Nvidia Pytorch image
FROM nvcr.io/nvidia/pytorch:22.06-py3

ENV TORCH_CUDA_ARCH_LIST="7.0;8.0"
ENV IABN_FORCE_CUDA=1
ENV FORCE_CUDA=1

# Copy files
COPY . .

# Install Python libraries
RUN python -m pip install -r requirements.txt

# Ready to run job with PyTorchJob and torchrun
ENTRYPOINT ["/bin/bash", "-c", "export PYTHONPATH=.;python trainer/train_boring_model.py; cat logs/lightning_logs/version_0/fit-run.pprof.txt"]
