# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Install OpenCV and other dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libfontconfig1 \
    ffmpeg \
    xvfb \
 && rm -rf /var/lib/apt/lists/*

# Install FFmpeg
RUN apt-get install -y ffmpeg

# Install any needed packages specified in requirements.txt
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Make port 2300 available to the world outside this container
EXPOSE 2300/udp

# Define environment variable
ENV NAME ObjectDetection

# Run app.py when the container launches
CMD ["python", "app.py"]