FROM python:3.10-bookworm

WORKDIR /backend

ENV FLASK_APP=backendInterface.py

COPY ./requirements.txt . 

RUN pip install -r requirements.txt

COPY . .

CMD ["python", "backendInterface.py"]