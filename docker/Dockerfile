FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV DISTRIBUTION=ubuntu2204

ARG PYTHON_V
ENV PYTHON_VERSION=$PYTHON_V

RUN apt-get update && \
    apt-get install -y --fix-broken --no-install-recommends \
        $PYTHON_VERSION \
        python3-distutils \
        python3-wheel \
        python3-pip && \
    # Clean up apt cache
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create symlink to always use $PYTHON_VERSION
RUN ln -s /usr/bin/$PYTHON_VERSION /usr/bin/python

# Install prod dependencies
COPY requirements-train.txt *.whl /tmp/
RUN python -m pip install pip -U && \
    pip install -r /tmp/requirements-train.txt --no-cache-dir && \
    pip install /tmp/*.whl --no-cache-dir && \
    rm /tmp/requirements-train.txt /tmp/*.whl 
