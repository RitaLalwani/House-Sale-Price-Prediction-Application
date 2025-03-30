# syntax=docker/dockerfile:1 
FROM python:3.12

# Set working directory
WORKDIR /app

# Copy files
COPY static/* /app
COPY templates/* /app
COPY functions/* /app
COPY . /app

# Install dependencies
RUN pip install -r requirements.txt

# Run the application
EXPOSE 8000
CMD python ./app.py
