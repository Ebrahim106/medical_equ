# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code and artifacts
COPY ml_api/ ml_api/
COPY artifacts/ artifacts/
COPY run.py .

# Set environment variables for production
ENV FLASK_DEBUG=false
ENV PORT=5000
# Ensure python output is not buffered
ENV PYTHONUNBUFFERED=1

# Expose the port
EXPOSE 5000

# Run the application using Gunicorn
CMD ["gunicorn", "--workers", "4", "--bind", "0.0.0.0:5000", "ml_api:create_app()"]
