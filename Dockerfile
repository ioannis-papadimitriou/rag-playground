# Use multi-stage build
FROM node:18 AS frontend-build

# Set working directory for frontend
WORKDIR /app/frontend

# Copy frontend package files
COPY rag-frontend/package*.json ./

# Install frontend dependencies
RUN npm install

# Copy frontend source
COPY rag-frontend/ ./

# Build frontend
RUN npm run build

# Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

# Copy Python requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code and cache
COPY . .
COPY .cache /app/.cache

# Copy built frontend from previous stage
COPY --from=frontend-build /app/frontend/dist /app/static

# Environment variables
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 5000

# Create necessary directories
RUN mkdir -p /app/uploads /app/sessions

# Set permissions
RUN chmod -R 755 /app/uploads /app/sessions

# Add a non-root user
RUN useradd -m myuser
RUN chown -R myuser:myuser /app
USER myuser

# Command to run the application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]