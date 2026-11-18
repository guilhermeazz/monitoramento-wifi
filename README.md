---
title: Wi-Fi CSI Spatial Radar
emoji: 📡
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Wi-Fi CSI Spatial Radar — Monitoramento Espacial sem Câmeras (MVP)

Sistema completo de **Localização 2D `(X, Y)`, Contagem de Pessoas em Tempo Real, Topologia Auto-Adaptativa (`1`, `2` ou `3+` sensores) e Inteligência Analítica via LLM**, utilizando **Roteador Wi-Fi 2.4 GHz (Cisco Linksys E1200) + Módulos ESP32-S3 + Notebook Servidor**.

---

## 1. Configuração e Teste do Servidor (Notebook)

### 1.1. Instalar Dependências e Iniciar o Servidor
Abra o PowerShell na pasta do projeto e execute:

```powershell
cd C:\Projetos\monitoramento1.0\wifi-spatial-mvp\backend

# (Opcional) Definir modelo padrão do Ollama (padrão: qwen2.5:7b; opções: gemma3:4b, llama3.2:3b)
# Você também pode trocar o modelo com 1 clique diretamente no seletor do Frontend!
$env:OLLAMA_MODEL="qwen2.5:7b"

# Iniciar o servidor FastAPI + WebSocket (10 Hz) + Receptor UDP (porta 5555)
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 1.1.1. Preparando o Ollama Local (100% Offline antes de conectar ao Roteador Isolado)
Antes de desconectar da internet e conectar na rede isolada do roteador **Cisco Linksys E1200**, baixe o modelo desejado no Ollama uma única vez:
```powershell
ollama pull qwen2.5:7b
# Ou alternativas mais leves:
ollama pull gemma3:4b
ollama pull llama3.2:3b
```
Quando o Ollama estiver aberto em segundo plano (`http://localhost:11434`), o painel exibirá automaticamente o badge verde **`LLM Local Online (Qwen 2.5)`** via `GET /api/llm/status`. Caso o Ollama esteja fechado, o sistema aciona automaticamente o **Fallback Estatístico Local** sem interromper a operação.

### 1.2. Testar o Painel Tático e a Topologia Auto-Adaptativa (Sem Hardware)
1. Abra o navegador em: **`http://localhost:8000`**
2. O sistema iniciará automaticamente no **Modo Simulador**:
   - No topo da tela, clique em **`1 ESP`**, **`2 ESPs`** ou **`3 ESPs`** para ver o **Motor de Topologia Auto-Adaptativa** alternar em tempo real entre:
     - **`1 ESP` (`SINGLE_LINK_CORRIDOR`):** Desenha a Elipse de Fresnel do corredor linear 1D e desloca o alvo sobre a reta `Roteador <-> ESP_1`.
     - **`2 ESPs` (`DUAL_LINK_SECTOR`):** Ativa interpolação vetorial setorial entre os 2 feixes.
     - **`3 ESPs` (`FULL_MESH_2D`):** Ativa triangulação 2D completa nos 4 quadrantes + Centro.
   - Clique nos botões **`0`**, **`1`**, **`2`**, **`3`**, **`4`** em *Pessoas* para testar a contagem e separação de alvos.
   - Arraste o ícone do **Roteador (`TX`)** ou dos **ESP32 (`RX`)** diretamente no mapa para testar o reposicionamento físico.
   - Clique em **"Calibrar Sala Vazia"** (coleta 5s de baseline) e depois em **"Gerar Relatório de Ações (LLM)"**.

### 1.3. Liberar a Porta UDP 5555 no Firewall do Windows (Essencial para Modo Hardware)
Para que o notebook receba os pacotes UDP vindos dos módulos ESP32-S3 na rede Wi-Fi, execute uma única vez no PowerShell (como Administrador):
```powershell
New-NetFirewallRule -DisplayName "WiFi CSI UDP 5555" -Direction Inbound -Protocol UDP -LocalPort 5555 -Action Allow
```

### 1.4. Descobrir o IP do Notebook na Rede do Roteador
Com o notebook conectado ao Wi-Fi do Cisco Linksys E1200, rode no PowerShell:
```powershell
ipconfig
```
Anote o **Endereço IPv4** da sua placa Wi-Fi (no Linksys E1200 por padrão será algo como `192.168.1.100` ou `192.168.1.102`). Você colocará esse IP na macro `SERVER_IP` do ESP32-S3.

---

## 2. Configuração Ideal do Roteador Cisco Linksys E1200 (2.4 GHz)

O **Cisco Linksys E1200** (Wireless-N 2.4 GHz MIMO 2x2) é excelente para extração de CSI porque opera nativamente em 802.11n na faixa de 2.4 GHz sem trocar de banda para 5 GHz (Band Steering).

1. Conecte-se ao Linksys E1200 e acesse o painel web: **`http://192.168.1.1`** (usuário padrão em branco ou `admin`, senha padrão `admin`).
2. Vá na aba **Wireless** -> **Basic Wireless Settings**:
   - **Configuration View:** selecione **Manual**.
   - **Network Mode:** selecione **Wireless-N Only** ou **Mixed** (importante manter o suporte a 802.11n ativo, pois os pacotes CSI de 52 subportadoras vêm dos preâmbulos `L-LTF` e `HT-LTF` do 802.11n).
   - **Network Name (SSID):** defina um nome simples sem espaços/acentos (ex: `CSI_RADAR_2G`).
   - **Channel Width:** selecione **20 MHz Only** (não deixe em *Auto 20/40 MHz*; travar em **20 MHz** garante estabilidade de fase e 52 subportadoras consistentes no ESP32-S3).
   - **Wide Channel / Standard Channel:** fixe um canal estático livre (ex: **Canal 1**, **Canal 6** ou **Canal 11**) em vez de `Auto`. Se o roteador mudar de canal sozinho no meio do monitoramento, a fase do CSI sofre salto.
   - Clique em **Save Settings**.
3. Vá na aba **Wireless** -> **Wireless Security**:
   - **Security Mode:** **WPA2 Personal** (AES).
   - Defina a **Passphrase** (senha que será colocada em `WIFI_PASS` no código do ESP32-S3).
4. Vá na aba **Wireless** -> **Advanced Wireless Settings**:
   - **AP Isolation:** certifique-se de que está em **Disabled** (desativado), para permitir que os módulos ESP32-S3 enviem pacotes UDP diretamente para o IP do seu Notebook na mesma WLAN.

---

## 3. Como Configurar e Gravar o Código no ESP32-S3

O código em [`firmware_esp32/main/main.c`](file:///c:/Projetos/monitoramento1.0/wifi-spatial-mvp/firmware_esp32/main/main.c) e o arquivo [`firmware_esp32/sdkconfig.defaults`](file:///c:/Projetos/monitoramento1.0/wifi-spatial-mvp/firmware_esp32/sdkconfig.defaults) já estão pré-configurados com `CONFIG_ESP_WIFI_CSI_ENABLED=y`.

### Passo 3.1: Editar Credenciais e ID do Sensor em `main.c`
Abra o arquivo `wifi-spatial-mvp/firmware_esp32/main/main.c` e edite as linhas 39 a 43:

```c
#define NODE_ID             "esp_1"            /* Use "esp_1" no 1º módulo, "esp_2" no 2º, "esp_3" no 3º */
#define WIFI_SSID           "CSI_RADAR_2G"     /* Nome da rede 2.4 GHz criada no Linksys E1200 */
#define WIFI_PASS           "SUA_SENHA_WIFI"   /* Senha do Wi-Fi do Linksys E1200 */
#define SERVER_IP           "192.168.1.100"    /* IPv4 do seu Notebook conectado ao Linksys E1200 */
#define SERVER_UDP_PORT     5555               /* Porta UDP padrão do Backend */
```

### Passo 3.2: Compilar e Gravar no ESP32-S3 (Terminal ESP-IDF v5.x)
1. Conecte a placa **ESP32-S3** ao notebook via cabo USB de dados (se a sua placa S3 tiver duas portas USB, use a porta marcada como **UART / COM** ou **USB**; verifique a porta `COMx` no Gerenciador de Dispositivos do Windows).
2. Abra o terminal **ESP-IDF 5.x PowerShell** (ou CMD) e execute:

```powershell
cd C:\Projetos\monitoramento1.0\wifi-spatial-mvp\firmware_esp32

# 1. Define o chip alvo especificamente para o ESP32-S3
idf.py set-target esp32s3

# 2. Confirma que o suporte a CSI foi habilitado pelo sdkconfig.defaults
# (Opcional: se quiser abrir o menu visual, rode `idf.py menuconfig` -> Component config -> Wi-Fi -> [*] WiFi CSI)

# 3. Compila, grava na porta COM da sua ESP32-S3 (ex: COM4) e abre o monitor serial
idf.py -p COM4 build flash monitor
```
*(Dica: Se a gravação ficar aguardando em `Connecting....`, segure o botão **BOOT** na placa ESP32-S3, aperte e solte o botão **RST/EN**, e solte o **BOOT**).*

### Passo 3.3: Verificar Funcionamento no Monitor Serial e no Painel Web
1. No monitor serial (`idf.py monitor`), você verá:
   - `Conectado! IP local: 192.168.1.105 | Gateway TX: 192.168.1.1`
   - `Captura de Wi-Fi CSI ativada com sucesso para NODE_ID=esp_1`
   - `Task geradora de tráfego RF iniciada (30 Hz).`
2. No navegador (`http://localhost:8000`), clique no botão superior **"Hardware Real (UDP:5555)"**:
   - Se você ligou **1 ESP32-S3** (`esp_1`), o sistema detectará automaticamente `1 ESP Ativo` e entrará no **Modo Corredor Linear 1D**!
   - Assim que ligar o 2º (`esp_2`) ou 3º (`esp_3`), o sistema mudará sozinho em tempo real para **Modo Setorial (2 ESPs)** ou **Malha 2D Completa (3 ESPs)**!
