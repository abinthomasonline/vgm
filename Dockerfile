FROM python:3.10.12-slim

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --upgrade pip==23.2.1 && \
    pip install -r requirements.txt

COPY . .

CMD ["python", "run.py"]
