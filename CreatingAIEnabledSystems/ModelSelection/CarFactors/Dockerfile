FROM python:3.10.11
WORKDIR /CarFactors
COPY ./requirements.txt /CarFactors
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8080
ENV FLASK_APP=carfactors_service.py
CMD ["python", "carfactors_service.py"]