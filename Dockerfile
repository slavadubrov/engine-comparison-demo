# syntax=docker/dockerfile:1
# Engine Comparison Demo - GPU-enabled container for distributed pipelines
# Supports: Spark, Ray, Daft with CUDA acceleration

FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    openjdk-17-jre-headless \
    curl \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Set JAVA_HOME for Spark
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"

# Install Apache Spark 4.1.1 (matches pyspark 4.1.1 in uv.lock)
ENV SPARK_VERSION=4.1.1
ENV SPARK_HOME=/opt/spark
RUN curl -fsSL https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop3.tgz \
    | tar -xz -C /opt \
    && mv /opt/spark-${SPARK_VERSION}-bin-hadoop3 /opt/spark
ENV PATH="${SPARK_HOME}/bin:${SPARK_HOME}/sbin:${PATH}"

# Install Hadoop AWS and AWS SDK JARs for S3A support (required for MinIO)
ENV HADOOP_VERSION=3.3.6
ENV AWS_SDK_VERSION=1.12.753
RUN curl -fsSL https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/${HADOOP_VERSION}/hadoop-aws-${HADOOP_VERSION}.jar \
    -o ${SPARK_HOME}/jars/hadoop-aws-${HADOOP_VERSION}.jar \
    && curl -fsSL https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/${AWS_SDK_VERSION}/aws-java-sdk-bundle-${AWS_SDK_VERSION}.jar \
    -o ${SPARK_HOME}/jars/aws-java-sdk-bundle-${AWS_SDK_VERSION}.jar

# Install uv for fast package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy project files
COPY pyproject.toml uv.lock ./
COPY src/ ./src/

# Install dependencies (including distributed extras)
RUN uv sync --frozen --extra distributed --extra notebook

# Make venv Python the default (must be AFTER uv sync creates it)
ENV PATH="/app/.venv/bin:${PATH}"
ENV PYTHONPATH="/app/src:${PYTHONPATH}"

# Configure AWS S3 endpoint for MinIO
ENV AWS_ENDPOINT_URL=http://minio:9000
ENV AWS_ACCESS_KEY_ID=minioadmin
ENV AWS_SECRET_ACCESS_KEY=minioadmin
ENV AWS_DEFAULT_REGION=us-east-1

# Default command
CMD ["bash"]
