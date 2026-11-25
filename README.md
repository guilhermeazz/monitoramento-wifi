---
title: Wi-Fi CSI Spatial Radar
emoji: 📡
colorFrom: slate
colorTo: sky
sdk: docker
app_port: 7860
pinned: false
---

# Monitoramento Espacial Wi-Fi CSI — ESP32-S3 + Cisco Linksys E1200

Sistema de **Sensoriamento Espacial sem Câmeras (Wi-Fi CSI)** com pipeline DSP vetorizado (**Sanitização HT20 + Normalização $L_2$ Anti-AGC + Seleção Automática NBVI + Filtro de Hampel & Butterworth + Segmentação MVS por Timestamp + Schmitt Trigger**), **Rastreamento Two-Stage Cascade (Kalman 2D)**, **Análise Contextual Assíncrona (LLM)** e interface **Dark Liquid Glass**.

---

## 1. Tabela Obrigatória de Configuração do Roteador Cisco Linksys E1200

Para garantir coerência de fase/amplitude do preâmbulo **802.11n HT20 (LLTF 64 subportadoras $\rightarrow$ 52 válidas)** e eliminar saltos espúrios de canal, configure o **Cisco Linksys E1200** (`http://192.168.1.1`) exatamente conforme a tabela abaixo:

| Seção no Painel Linksys E1200 | Parâmetro | Valor Obrigatório | Motivo Técnico (Física de RF / CSI) |
| :--- | :--- | :--- | :--- |
| **Wireless > Basic Wireless Settings** | `Configuration View` | **Manual** | Permite travar canal e largura de banda fixos. |
| **Wireless > Basic Wireless Settings** | `Network Mode` | **Wireless-N Only** | Garante modulação OFDM 802.11n com preâmbulos `L-LTF` e `HT-LTF` em todos os pacotes. |
| **Wireless > Basic Wireless Settings** | `Channel Width` | **20 MHz Only** | Fixa o vetor CSI em 64 subportadoras (128 bytes I/Q). Nunca use *Auto (20/40 MHz)*. |
| **Wireless > Basic Wireless Settings** | `Standard Channel` | **Canal Fixo 1, 6 ou 11** | Impede *Auto Channel Hopping*, que alteraria a frequência central e invalidaria a calibração NBVI. |
| **Wireless > Basic Wireless Settings** | `SSID` | **`CSI_RADAR_2G`** (Rede Dedicada) | Isola o tráfego de sensoriamento de celulares de terceiros. |
| **Wireless > Wireless Security** | `Security Mode` | **WPA2 Personal (AES)** | Criptografia rápida em hardware sem fallback TKIP legado. |
| **Wireless > Advanced Wireless** | `AP Isolation` | **Disabled** | Permite tráfego UDP direto dos nós ESP32-S3 para o servidor. |
| **Conexão Física do Servidor** | Porta **Ethernet LAN (1–4)** | **Cabo RJ45 (Cat5e/Cat6)** | Conectar o servidor via cabo Ethernet elimina colisão no meio sem fio e reduz a latência UDP para $< 2\text{ ms}$. |

---

## 2. Guia de Posicionamento Físico e Protocolo de Validação Gate 0

### 2.1. Geometria e Instalação dos Sensores ESP32-S3
1. **Distância TX–RX (`3.0 m` a `6.0 m`):** Posicione o roteador **Cisco Linksys E1200 (TX)** e cada placa **ESP32-S3 (RX)** a uma distância entre **$3\text{ m}$ e $6\text{ m}$**. Distâncias $< 1.5\text{ m}$ saturam o LNA com a onda direta (LoS), enquanto distâncias $> 8\text{ m}$ degradam a relação sinal-ruído (SNR).
2. **Altura do Solo (`1.0 m` a `1.2 m`):** Instale tanto o roteador quanto os nós ESP32-S3 na altura do tronco humano (**$1.0\text{ m}$ a $1.2\text{ m}$** do piso), maximizando a seção reta radar (RCS) do tórax dentro da **1ª Zona de Fresnel**.
3. **Orientação da Antena PCB:** Mantenha a placa ESP32-S3 com a **antena PCB na vertical**, afastada pelo menos **$20\text{ cm}$** de superfícies metálicas, monitores, gabinetes ou paredes de concreto armado.
4. **Abertura Angular Multi-Nó (`60°` a `120°`):**
   - **1 Nó:** Crie um corredor linear cruzando a área de passagem principal.
   - **2 Nós:** Posicione os dois ESP32-S3 formando um ângulo de **$60^\circ$ a $90^\circ$** a partir do roteador para separar os setores `ZONA_A`, `ZONA_B` e `INTERSECAO`.
   - **3+ Nós:** Distribua os sensores ao redor do perímetro com separação angular entre **$60^\circ$ e $120^\circ$**, formando uma malha cruzada de elipses de Fresnel para localização 2D via Filtro de Kalman.

### 2.2. Protocolo de Validação de Campo — Gate 0
Antes de iniciar a operação oficial em um novo ambiente, valide a instalação física executando o protocolo **Gate 0**:
1. Execute a **Calibração Automática Guiada** (`5 s` de saída + `10 s` de captura com a sala vazia).
2. Verifique se o badge de qualidade resulta em **`OPTIMAL`** ou **`ACCEPTABLE`** (taxa de pacotes $\ge 30\text{ Hz}$).
3. Anote o valor basal de turbulência em sala vazia ($\text{MVS}_{\text{vazio}}$) exibido no card *Sinal em Tempo Real*.
4. Peça para 1 pessoa caminhar normalmente cruzando a linha de visada TX–RX e observe o pico $\text{MVS}_{\text{movimento}}$:
   $$\frac{\text{MVS}_{\text{movimento}}}{\text{MVS}_{\text{vazio}}} \ge 4.0$$
   - Se a razão for $\ge 4.0$, o **Gate 0 está aprovado** (margem superior a $300\%$ acima do limiar $\theta_{\text{on}} = 1.25 \cdot P_{95}(\text{MVS}_{\text{vazio}})$).

---

## 3. Como Executar o Servidor, Testes e Gravar o Firmware ESP32-S3

### 3.1. Executar o Backend + Interface Dark Liquid Glass
```powershell
cd C:\Projetos\monitoramento1.0\wifi-spatial-mvp\backend

# Instalar dependências
pip install -r requirements.txt

# Iniciar o servidor FastAPI (HTTP/WebSocket na porta 8000 + Receptor UDP Binário 0xC510 na porta 5555)
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```
Acesse no navegador: **`http://localhost:8000`**

### 3.2. Executar a Suíte de Testes Automatizados (`pytest`)
```powershell
cd C:\Projetos\monitoramento1.0\wifi-spatial-mvp
python -m pytest backend/tests/ -v
```

### 3.3. Compilar e Gravar o Firmware no ESP32-S3 (ESP-IDF v5.x / v6.x)
Edite as configurações em [`firmware_esp32/main/csi_config.h`](file:///c:/Projetos/monitoramento1.0/wifi-spatial-mvp/firmware_esp32/main/csi_config.h) (ou via `idf.py menuconfig` no menu *Wi-Fi CSI Spatial Sensing Configuration*) definindo `CSI_NODE_ID_NUM` (`1`, `2` ou `3`), `CSI_WIFI_SSID`, `CSI_WIFI_PASS` e `CSI_SERVER_IP`. Em seguida:

```powershell
cd C:\Projetos\monitoramento1.0\wifi-spatial-mvp\firmware_esp32
idf.py set-target esp32s3
idf.py -p COM4 build flash monitor
```
