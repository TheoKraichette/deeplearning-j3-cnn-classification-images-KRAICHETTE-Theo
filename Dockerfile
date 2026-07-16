FROM tensorflow/tensorflow:2.18.0-jupyter

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
