FROM mambaorg/micromamba:1.4.3
USER root
ARG MAMBA_DOCKERFILE_ACTIVATE=1
WORKDIR /workspace
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -r /var/lib/apt/lists/*
RUN micromamba install --yes --name base --channel conda-forge \
      pyopenssl  \
      python=3.9.1 \
      requests=2.25.1 \
      numcodecs \
      imagecodecs \
      psutil && \
    micromamba clean --all --yes && \
    pip3 install \
      torch \
      torchvision \
      torchaudio \
      --index-url https://download.pytorch.org/whl/cpu

RUN git clone https://github.com/Project-MONAI/MONAI.git \
    && cd ./MONAI \
# note: the dev branch is the default branch after cloning MONAI
    && pip install . \
    && cd .. \
    && git clone https://github.com/MMV-Lab/mmv_im2im.git \
    && cd ./mmv_im2im \
    && pip install -e .[all] \
    && cd ..
WORKDIR /workspace/mmv_im2im