# syntax=docker/dockerfile:1

# Start from the Nvidia Pytorch image
FROM nvcr.io/nvidia/pytorch:22.06-py3

# Set-up user and env variables
ARG USER_ID
ARG USER_NAME

RUN useradd -m --no-log-init --system  --uid ${USER_ID} ${USER_NAME} -g users && \
    usermod -a -G sudo ${USER_NAME} && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

USER ${USER_NAME}:users

ENV TORCH_CUDA_ARCH_LIST="7.0;8.0"
ENV IABN_FORCE_CUDA=1
ENV FORCE_CUDA=1

# Copy files
WORKDIR /home/${USER_NAME}
COPY --chown=${USER_NAME}:users . .


# Install Python libraries
RUN python -m pip install -r requirements.txt

# Ready to run job with PyTorchJob and torchrun
ENTRYPOINT ["/bin/bash", "-c", "export PYTHONPATH=.;python trainer/train_boring_model.py; cat logs/lightning_logs/version_0/fit-run.pprof.txt"]
