FROM python:3.7

WORKDIR /app

COPY . .

RUN pip install --upgrade pip setuptools wheel

RUN pip install -r requirements.txt

CMD ["python3", "app.py"]

