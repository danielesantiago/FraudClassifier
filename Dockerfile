FROM python:3.12-slim

RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PYTHONPATH="${PYTHONPATH}:/app/src"

# Copia arquivos para o container
COPY requirements.txt .
COPY app/ app/
COPY src/ src/

# Instala dependências
RUN pip install --no-cache-dir -r requirements.txt

# Expõe a porta da API
EXPOSE 8000

# Comando para iniciar a API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
