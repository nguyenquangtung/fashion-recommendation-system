# Use the official Python image with tag 3.11
FROM python:3.11

# Set the working directory in the container
WORKDIR /app

# Copy the dependencies file to the working directory
COPY requirement.txt .

# Install any needed packages specified in requirements.txt
RUN pip install -r requirement.txt

# Copy the content of the local src directory to the working directory
COPY . .

# Specify the command to run on container start
CMD ["python", "API.py"]

