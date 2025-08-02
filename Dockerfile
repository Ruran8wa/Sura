FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Use environment variable PORT that Railway sets
ENV PORT 8000

EXPOSE $PORT

# Start uvicorn with host 0.0.0.0 and the port Railway provides
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
