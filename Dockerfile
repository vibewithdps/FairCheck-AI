# Use the official Python image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy all local files to the container
COPY . .

# Install the dependencies from your requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Streamlit uses port 8501 by default, but Cloud Run expects 8080
EXPOSE 8080

# Command to run the app on the specific port Cloud Run requires
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]