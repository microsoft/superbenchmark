FROM nvcr.io/nvidia/pytorch:25.08-py3@sha256:b70bc3ff73fae58e7ec326849b4726eac04200d9ec1da2b385f94c9f7775971f

COPY dockerfile/requirements/deepseek-v2-lite-reference.txt /tmp/deepseek-v2-lite-reference.txt
RUN python3 -m venv --system-site-packages /opt/deepseek-reference && \
    /opt/deepseek-reference/bin/pip install --no-cache-dir -r /tmp/deepseek-v2-lite-reference.txt
