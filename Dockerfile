FROM python:3.11-slim

WORKDIR /app

# Instalează dependențe sistem (pentru fontconfig și fonts)
RUN apt-get update && apt-get install -y \
    fontconfig \
    fonts-dejavu \
    && rm -rf /var/lib/apt/lists/*

# Copiază requirements
COPY requirements.txt .

# Instalează dependențe Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiază aplicația
COPY . .

# Expune portul
EXPOSE 8000

# Comandă de start - AJUSTEAZĂ după aplicația ta
CMD ["python", "app.py"]
