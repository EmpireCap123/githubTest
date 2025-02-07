#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Example Dockerfile (adapt to your needs)
FROM python:3.9-slim-buster  # Use a suitable base image

WORKDIR /app

COPY requirements.txt .  # Copy requirements first for caching
RUN pip install -r requirements.txt

COPY . .  # Copy the rest of your application code

CMD ["python", "your_app.py"]  # Command to run when the container starts

