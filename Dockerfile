# Use a base Python image
FROM python:3.9

# Set the desired base directory path
ARG BASE_DIR=/app

# Set the working directory inside the container
WORKDIR ${BASE_DIR}

# Install system dependencies
RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 python3-lxml

# Copy the requirements.txt file
COPY requirements.txt .

# Install dependencies from requirements.txt
RUN pip install -r requirements.txt

# Set the entrypoint script
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Set the base directory as an environment variable
ENV BASE_DIR=${BASE_DIR}

# Update the entrypoint script to use the environment variable
ENTRYPOINT ["./entrypoint.sh", "$BASE_DIR"]
