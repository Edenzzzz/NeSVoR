# syntax=docker/dockerfile:1

# source of tinycudann pytorch binding binaries.
# See https://github.com/FNNDSC/tinycudann-docker-build
FROM docker.io/fnndsc/tinycudann:isolate-python3.10.6-pytorch1.13.1-cuda11.7 as tinycudann-isolate

FROM docker.io/mambaorg/micromamba:1.4-focal-cuda-11.7.1 AS dependency-installer

COPY --chown=$MAMBA_USER:$MAMBA_USER env.yaml /tmp/env.yaml

# uid, gid are hard-coded from the base image mambaorg/micromamba
RUN --mount=type=cache,target=/opt/conda/pkgs,uid=1000,gid=1000,sharing=locked \
    micromamba install -y -n base -f /tmp/env.yaml

COPY --from=tinycudann-isolate /usr/local/lib/python3.10/site-packages/ /opt/conda/lib/python3.10/site-packages/

FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04 AS builder
COPY --from=dependency-installer /opt/conda /opt/conda
ENV PATH=/opt/conda/bin:$PATH LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH

WORKDIR /tmp/src/nesvor
COPY ./nesvor/ ./nesvor/
COPY ./setup.py ./setup.py

# TORCH_CUDA_ARCH_LIST is a list of NVIDIA GPU compute capability numbers to target.
# https://github.com/pytorch/pytorch/blob/d922c29a22e4bf0fba49526f7536395eb8cd66f4/torch/utils/cpp_extension.py#L975-L998
# The value here must be a subset of the values of TCNN_CUDA_ARCHITECTURES which was used to build tiny-cuda-nn
# Note: compute capability architectures 9.0 8.9 5.2 3.7 do not work.
ARG TORCH_CUDA_ARCH_LIST="8.6 8.0 7.5 7.0 6.1+PTX"
RUN pip install --verbose .

# FIXME: TEMPORARY HACK
# A better way to install weights: use package_data. Example:
# https://github.com/FNNDSC/pl-covidnet-pdfgeneration/blob/038ef4e9bb8f2d6530b96622a12826d77ced0bd7/setup.py#L22-L24
# https://github.com/FNNDSC/pl-covidnet-pdfgeneration/blob/038ef4e9bb8f2d6530b96622a12826d77ced0bd7/pdfgeneration/pdfgeneration.py#L101
COPY --from=junshenxu/nesvor:v0.5.0 /usr/local/NeSVoR/nesvor/checkpoints/ /opt/conda/lib/python3.10/site-packages/nesvor/checkpoints/

FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04
COPY --from=builder /opt/conda /opt/conda
ENV PATH=/opt/conda/bin:$PATH LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH
