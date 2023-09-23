FROM python:3.10.9
WORKDIR /app
COPY . /app

RUN apt update -y && apt install awscli -y 
EXPOSE "8000"
RUN pip install -r requirements.txt
CMD [ "python", "app.py" ]
