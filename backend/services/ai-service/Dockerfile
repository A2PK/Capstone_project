# Use the official Python image
FROM python:3.12.10-slim

# Install curl and Poetry
RUN apt-get update && apt-get install -y curl \
    && curl -sSL https://install.python-poetry.org | python3 - \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Add Poetry to PATH
ENV PATH="/root/.local/bin:$PATH"

# Set work directory
WORKDIR /app

# Copy only poetry files first for layer caching
COPY services/ai-service/pyproject.toml services/ai-service/poetry.lock ./

# Install dependencies
RUN poetry install --no-interaction --no-root

# Copy the rest of the application
COPY services/ai-service/ .

# Expose the port the app runs on
EXPOSE 8000

# Start FastAPI with uvicorn
CMD ["poetry", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]