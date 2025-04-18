# Use the official Python image
FROM python:3.12.10-slim

# Set work directory
WORKDIR /app

# Copy files
COPY services/ai-service/ .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port the app runs on
EXPOSE 8000

# CMD ["python", "main.py"]
# Start FastAPI with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]