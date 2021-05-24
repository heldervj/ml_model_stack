FROM python:latest

COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
RUN chmod +x ./run.sh

ENTRYPOINT ["./run.sh"]