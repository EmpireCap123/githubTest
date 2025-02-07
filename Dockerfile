FROM python:3.9-slim  # Use a slim Python base image

WORKDIR /app  # Set the working directory inside the container

COPY requirements.txt .  # Copy requirements file (if any)
RUN pip install --no-cache-dir -r requirements.txt  # Install dependencies

COPY data_update.py .  # Copy your Python script

CMD ["python", "data_update.py"]  # Command to run your script
