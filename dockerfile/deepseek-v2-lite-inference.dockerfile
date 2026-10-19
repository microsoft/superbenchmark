FROM vllm/vllm-openai:v0.31.0@sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3

COPY dockerfile/requirements/deepseek-v2-lite-reference.txt /tmp/deepseek-v2-lite-reference.txt
RUN python3 -m venv --system-site-packages /opt/deepseek-reference && \
    python3 -c "import pathlib, sys, sysconfig; source=sysconfig.get_path('purelib'); target=pathlib.Path('/opt/deepseek-reference/lib') / ('python%d.%d' % sys.version_info[:2]) / 'site-packages'; (target / 'engine-torch.pth').write_text(source + '\n')" && \
    /opt/deepseek-reference/bin/pip install --no-cache-dir -r /tmp/deepseek-v2-lite-reference.txt
