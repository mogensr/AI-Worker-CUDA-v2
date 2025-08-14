FROM nvidia/cuda:11.8-devel-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python3", "handler.py"]
