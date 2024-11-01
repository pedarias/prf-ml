# Utilizar uma imagem base com Python 3.12
FROM python:3.12-slim

# Definir o diretório de trabalho dentro do container
WORKDIR /app

# Copiar o arquivo requirements.txt para o diretório de trabalho
COPY requirements.txt .

# Instalar as dependências
RUN pip install --no-cache-dir -r requirements.txt

# Copiar os diretórios e arquivos necessários para o diretório de trabalho
# Isso inclui o diretório src e possivelmente outros arquivos ou diretórios como data, LICENSE, etc.
COPY src/ .
COPY data/ data/
COPY images/ images/
COPY artifacts/ artifacts/
COPY classes-com-peso/ classes-com-peso/
COPY classes-sem-peso/ classes-sem-peso/
COPY datatran2024.csv .
COPY README.md .
COPY LICENSE .

# Expor a porta que a aplicação irá rodar
EXPOSE 8000

# Comando para iniciar a aplicação usando o Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
