FROM tensorflow/tensorflow:2.16.1-gpu
WORKDIR /home/user/work
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt