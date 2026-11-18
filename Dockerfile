FROM python:3.11-slim

# Cria o usuário não-root (UID 1000) exigido pelo Hugging Face Spaces
RUN useradd -m -u 1000 user

WORKDIR /app

# Instala dependências Python
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/backend/requirements.txt

# Copia backend e frontend garantindo permissões de leitura/escrita para o usuário 1000
COPY --chown=user:user backend/ /app/backend/
COPY --chown=user:user frontend/ /app/frontend/

# Alterna para o usuário não-root
USER user

# Configurações para Hugging Face Spaces:
# - INITIAL_MODE=simulation: Inicia já no modo simulado para qualquer visitante testar
# - INITIAL_SIM_PEOPLE=1: Já exibe 1 pessoa se movimentando suavemente no radar
# - INITIAL_SIM_NODES=3: Malha de 3 ESPs (Full Mesh 2D)
# - HTTP_PORT=7860: Porta padrão exposta pelo Hugging Face Spaces
ENV PYTHONUNBUFFERED=1 \
    INITIAL_MODE=simulation \
    INITIAL_SIM_PEOPLE=1 \
    INITIAL_SIM_NODES=3 \
    HOST=0.0.0.0 \
    HTTP_PORT=7860

EXPOSE 7860

WORKDIR /app/backend
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
