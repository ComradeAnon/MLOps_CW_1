FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY model.joblib .
CMD ["python", "-c", "import joblib; print(joblib.load('model.joblib'))"]
