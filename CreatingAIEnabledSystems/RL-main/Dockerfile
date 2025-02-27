FROM python:3.8-slim

WORKDIR /usr/src/app

COPY . .

RUN pip3 install -r requirements.txt

EXPOSE 8080

CMD ["python", "app.py"]