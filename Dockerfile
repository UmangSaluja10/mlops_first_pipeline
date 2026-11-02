# Pull the python image from the docker hub
# Use a lightweight Python image
FROM python:3.14-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install -r requirements.txt

# Copy the rest of the application code and saved models into the container
COPY . .
# first dot: source, second dot: destination

# Document the port the app runs on
EXPOSE 8000

# Command to run the application
# 0.0.0.0 allows the app to be accessible from outside the container
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]