FROM vllm/vllm-openai:v0.31.0@sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3

RUN apt-get update && apt-get install -y --no-install-recommends git && apt-get clean

COPY dockerfile/requirements/deepseek-v2-lite-controller.txt /tmp/deepseek-v2-lite-controller.txt
RUN mkdir -p /opt/vllm/bin && ln -s /usr/bin/python3 /opt/vllm/bin/python && \
    python3 -m venv --system-site-packages /opt/deepseek-reference && \
    python3 -c "import pathlib, sys, sysconfig; source=sysconfig.get_path('purelib'); target=pathlib.Path('/opt/deepseek-reference/lib') / ('python%d.%d' % sys.version_info[:2]) / 'site-packages'; (target / 'engine-torch.pth').write_text(source + '\n')" && \
    /opt/deepseek-reference/bin/pip install --no-cache-dir --no-deps -r /tmp/deepseek-v2-lite-controller.txt

WORKDIR /opt/superbench
COPY setup.py MANIFEST.in LICENSE README.md ./
COPY superbench ./superbench
RUN /opt/deepseek-reference/bin/pip install --no-cache-dir --no-deps --no-build-isolation . && \
    /opt/deepseek-reference/bin/python -c "import numpy, torch, transformers; assert transformers.__version__ == '4.46.3'; assert torch.from_numpy(numpy.ones(1, dtype=numpy.float32)).numpy()[0] == 1" && \
    /opt/vllm/bin/python -c "from importlib.metadata import version; import numpy, torch; assert version('vllm').split('+')[0] == '0.31.0'; assert version('transformers') == '5.17.0'; assert torch.from_numpy(numpy.ones(1, dtype=numpy.float32)).numpy()[0] == 1" && \
    /opt/deepseek-reference/bin/sb run --help > /dev/null && \
    /opt/deepseek-reference/bin/sb result diagnosis --help > /dev/null

RUN apt-get update && apt-get install -y --no-install-recommends rsync && apt-get clean

ENV PATH="/opt/deepseek-reference/bin:${PATH}"
ENTRYPOINT ["sb"]
