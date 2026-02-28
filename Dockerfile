# 1. Base Image: Use a lightweight Python version
FROM python:3.9-slim

# 2. System Settings: Don't write .pyc files, and print logs immediately
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Create a work directory
WORKDIR /app

# 4. Install dependencies
# We copy requirements first to cache them (makes rebuilding faster)
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the entire project code
COPY . /app/

# 6. Expose the port Streamlit runs on
EXPOSE 8501

# 7. The Command to run your App
CMD ["streamlit", "run", "app/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]