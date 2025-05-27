# STEP 1: Base image with Python 3.10 (can be 3.11 if needed)
FROM python:3.11-slim

# STEP 2: Set working directory inside the container
WORKDIR /app

# STEP 3: Copy your project files into the container
COPY . .

# STEP 4: Install dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# STEP 5: Default command (can override later)
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "2000"]