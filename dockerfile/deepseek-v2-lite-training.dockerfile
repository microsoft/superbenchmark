FROM nvcr.io/nvidia/pytorch:25.08-py3@sha256:b70bc3ff73fae58e7ec326849b4726eac04200d9ec1da2b385f94c9f7775971f

COPY dockerfile/requirements/deepseek-v2-lite-controller.txt /tmp/deepseek-v2-lite-controller.txt
RUN python3 -m venv --system-site-packages /opt/deepseek-reference && \
    /opt/deepseek-reference/bin/pip install --no-cache-dir --no-deps -r /tmp/deepseek-v2-lite-controller.txt

WORKDIR /opt/superbench
COPY setup.py MANIFEST.in LICENSE README.md ./
COPY superbench ./superbench
RUN /opt/deepseek-reference/bin/pip install --no-cache-dir --no-deps --no-build-isolation . && \
    /opt/deepseek-reference/bin/python -c "import numpy, torch, transformers; assert transformers.__version__ == '4.46.3'; assert torch.from_numpy(numpy.ones(1, dtype=numpy.float32)).numpy()[0] == 1" && \
    /opt/deepseek-reference/bin/sb run --help > /dev/null && \
    /opt/deepseek-reference/bin/sb result diagnosis --help > /dev/null

RUN apt-get update && apt-get install -y --no-install-recommends rsync && apt-get clean

ENV PATH="/opt/deepseek-reference/bin:${PATH}"
ENTRYPOINT ["sb"]
