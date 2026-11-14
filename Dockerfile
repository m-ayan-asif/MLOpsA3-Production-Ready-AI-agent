FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY requirements-inference.txt .
RUN pip install -r requirements-inference.txt
COPY api/ api/
COPY models/model.joblib /models/model.joblib
RUN useradd -r -u 10001 appuser
USER 10001
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
