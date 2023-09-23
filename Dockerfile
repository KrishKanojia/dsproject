FROM python:3.10.9
WORKDIR /app
COPY . /app
EXPOSE 8080

RUN apt update -y

RUN apt-get update && pip install -r requirements.txt

CMD [ "python", "app.py" ]
