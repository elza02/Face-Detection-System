FROM openjdk:17-slim

# ps command
RUN apt-get update && apt-get install -y procps

# Install build dependencies
RUN apt-get install -y \
    wget \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl

# Download and install Python 3.10.0
RUN wget https://www.python.org/ftp/python/3.10.0/Python-3.10.0.tgz && \
    tar xzf Python-3.10.0.tgz && \
    cd Python-3.10.0 && \
    ./configure --enable-optimizations && \
    make altinstall && \
    cd .. && \
    rm -rf Python-3.10.0 Python-3.10.0.tgz

# Update alternatives to make Python 3.10 the default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.10 1 && \
    update-alternatives --set python3 /usr/local/bin/python3.10

# Install pip for Python 3.10
RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3.10 get-pip.py && \
    rm get-pip.py

# Set up Spark
ENV SPARK_VERSION=3.4.3
ENV HADOOP_VERSION=3
ENV SPARK_HOME=/opt/spark

# Add more robust download with multiple mirrors and error checking
RUN set -ex; \
    SPARK_URL="https://dlcdn.apache.org/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz"; \
    BACKUP_URL="https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz"; \
    wget -t 3 -T 30 -q --show-progress -O spark.tgz "$SPARK_URL" || \
    wget -t 3 -T 30 -q --show-progress -O spark.tgz "$BACKUP_URL" || \
    (echo "Failed to download Spark" && exit 1); \
    tar xzf spark.tgz; \
    mv spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} ${SPARK_HOME}; \
    rm spark.tgz

ENV PATH=$SPARK_HOME/bin:$PATH

# Download Kafka connector for Spark
RUN mkdir -p /app/jars && \
    wget -q -O /app/jars/spark-sql-kafka-0-10_2.12-3.4.3.jar https://repo1.maven.org/maven2/org/apache/spark/spark-sql-kafka-0-10_2.12/3.4.3/spark-sql-kafka-0-10_2.12-3.4.3.jar && \
    wget -q -O /app/jars/kafka-clients-3.4.0.jar https://repo1.maven.org/maven2/org/apache/kafka/kafka-clients/3.4.0/kafka-clients-3.4.0.jar && \
    wget -q -O /app/jars/kafka-streams-3.4.0.jar https://repo1.maven.org/maven2/org/apache/kafka/kafka-streams/3.4.0/kafka-streams-3.4.0.jar && \
    wget -q -O /app/jars/kafka-streams-scala_2.12-3.4.0.jar https://repo1.maven.org/maven2/org/apache/kafka/kafka-streams-scala_2.12/3.4.0/kafka-streams-scala_2.12-3.4.0.jar

# Verify that the JAR files are downloaded correctly
RUN ls -l /app/jars

WORKDIR /app

COPY requirements.txt .
RUN #pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

ENV PYSPARK_PYTHON=python3.10
ENV PYSPARK_DRIVER_PYTHON=python3.10
ENV SPARK_CLASSPATH="/app/jars/*"

#CMD ["spark-submit", "--jars", "/app/jars/spark-sql-kafka-0-10_2.12-3.4.3.jar,/app/jars/kafka-clients-2.8.2.jar,/app/jars/spark-token-provider-kafka-0-10_2.12-3.4.3.jar,/app/jars/commons-pool2-2.11.1.jar", "--conf", "spark.executor.extraClassPath=/app/jars/*", "--conf", "spark.driver.extraClassPath=/app/jars/*", "app.py"]
CMD ["spark-submit", "--jars", "/app/jars/spark-sql-kafka-0-10_2.12-3.4.3.jar,/app/jars/kafka-clients-3.4.0.jar,/app/jars/kafka-streams-3.4.0.jar,/app/jars/kafka-streams-scala_2.12-3.4.0.jar", "--conf", "spark.executor.extraClassPath=/app/jars/*", "--conf", "spark.driver.extraClassPath=/app/jars/*", "app.py"]
