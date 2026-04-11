FROM python:3.10-slim

WORKDIR /app

# Install server runtime dependencies only.
COPY requirements-server.txt .
RUN pip install --no-cache-dir -r requirements-server.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 7860

# Run the OpenEnv-compatible server
CMD ["python", "-m", "server.app", "--port", "7860"]
