FROM python:3.10-buster
COPY setup.py setup.py
COPY requirements.txt requirements.txt
COPY best-n.pt best.pt
RUN pip install --upgrade pip
RUN pip install -e .

# libraries required by OpenCV (working with images)
RUN apt-get update && \
apt-get install \
  'tesseract-ocr' \
  'libtesseract-dev' \
  'ffmpeg'\
  'libsm6'\
  'libxext6'  -y

COPY donutplot donutplot/
CMD uvicorn donutplot.api.fast:app --host 0.0.0.0 --port $PORT
