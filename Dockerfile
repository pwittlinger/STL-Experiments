FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y git
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*
RUN python -m pip install "git+https://github.com/StanfordASL/stlcg"

RUN apt-get update && apt-get install -y graphviz

COPY requirements.txt ./
COPY . /app
#WORKDIR /app
ENV PATH="$PATH:/Graphviz/bin"
RUN pip install -r requirements.txt --no-cache-dir
#ENTRYPOINT [ "python" ]
CMD ["python", "./stlExperiments.py"]
