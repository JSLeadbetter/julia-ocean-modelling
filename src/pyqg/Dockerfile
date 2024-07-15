# User Ubuntu as the base.
FROM --platform=linux/amd64 ubuntu:24.04

# Ensure things are up to date.
RUN apt-get update && apt-get install -y --no-install-recommends

# Install essentials including gcc.
RUN apt install build-essential -y

# Install curl
RUN apt install curl -y
RUN apt install wget -y

# Download FFTW.
RUN curl http://www.fftw.org/fftw-3.3.10.tar.gz -o fftw-3.3.10.tar.gz && \
    tar -xf fftw-3.3.10.tar.gz && \
    rm fftw-3.3.10.tar.gz 

# Install FFTW.
RUN cd fftw-3.3.10 && \ 
    ./configure && \
    gmake && \
    gmake install

# Install Python.
RUN apt-get install -y python3

# Download Conda.
RUN mkdir -p ~/miniconda3 && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh

# Install Conda.
ENV CONDA_DIR /opt/conda
RUN bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3 && \
    rm -rf ~/miniconda3/miniconda.sh

# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH

RUN apt-get install zsh git fonts-font-awesome -y

RUN zsh

# Uses "robbyrussell" theme (original Oh My Zsh theme), with no plugins
RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.2.0/zsh-in-docker.sh)" -- \
    -t robbyrussell
