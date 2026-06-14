# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Create a non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set the working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code and artifacts
COPY ml_api/ ml_api/
COPY artifacts/ artifacts/
COPY run.py .
COPY gunicorn.conf.py .

# Change ownership to the non-root user
RUN chown -R appuser:appuser /app

# Switch to the non-root user
USER appuser

# Set environment variables for production
ENV FLASK_DEBUG=false
ENV PORT=5000
# Ensure python output is not buffered
ENV PYTHONUNBUFFERED=1

# Expose the port
EXPOSE 5000

# Run the application using Gunicorn with the config file
CMD ["gunicorn", "-c", "gunicorn.conf.py", "ml_api:create_app()"]
