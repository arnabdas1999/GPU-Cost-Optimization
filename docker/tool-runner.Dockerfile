# Image for public-test and grading containers. Build on the GPU host:
#   docker build -t rollout-cost-tools:py312 -f docker/tool-runner.Dockerfile docker
# The runner records the resulting image id in every run manifest.
FROM python:3.12-slim
RUN pip install --no-cache-dir pytest==8.3.3
USER 65534:65534
WORKDIR /workspace
