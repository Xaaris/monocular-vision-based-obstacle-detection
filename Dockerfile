FROM python:3.7
COPY . /
RUN pip install --upgrade pip setuptools
RUN pip install -r requirements.txt
CMD python ./Main.py