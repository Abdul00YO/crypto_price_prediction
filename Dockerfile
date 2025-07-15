# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy project files to container
COPY . .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip \
 && pip install -r requirements.txt

# Streamlit settings
ENV PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ENABLECORS=false

# Expose Streamlit port
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
