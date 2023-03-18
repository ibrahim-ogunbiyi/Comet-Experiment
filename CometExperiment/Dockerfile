FROM python:slim
WORKDIR /code
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
CMD ["python", "./CometExperiment.py"]