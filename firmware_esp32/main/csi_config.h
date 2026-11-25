/**
 * ============================================================================
 * csi_config.h — Configuração Centralizada do Nó Sensor ESP32-S3 Wi-Fi CSI
 * Projeto: Monitoramento Espacial Wi-Fi CSI (Cisco Linksys E1200 + ESP32-S3)
 * ============================================================================
 */
#ifndef CSI_CONFIG_H
#define CSI_CONFIG_H

#include <stdint.h>
#include <stdbool.h>
#if defined(__has_include)
#  if __has_include("sdkconfig.h")
#    include "sdkconfig.h"
#  endif
#endif

/* ----------------------------------------------------------------------------
 * 1. Identificação do Nó e Credenciais de Rede
 *    CSI_USE_KCONFIG = 0 faz com que os valores definidos aqui tenham prioridade
 *    absoluta sobre o cache do `sdkconfig` (ideal para gravar esp_1, esp_2, esp_3).
 * ---------------------------------------------------------------------------- */
#define CSI_USE_KCONFIG         0

#if CSI_USE_KCONFIG && defined(CONFIG_CSI_NODE_ID)
#define CSI_NODE_ID_NUM         CONFIG_CSI_NODE_ID
#else
#define CSI_NODE_ID_NUM         1               /* Altere para 1, 2 ou 3 conforme a placa */
#endif

#if CSI_USE_KCONFIG && defined(CONFIG_CSI_WIFI_SSID)
#define CSI_WIFI_SSID           CONFIG_CSI_WIFI_SSID
#else
#define CSI_WIFI_SSID           "CSI_RADAR_2G"
#endif

#if CSI_USE_KCONFIG && defined(CONFIG_CSI_WIFI_PASSWORD)
#define CSI_WIFI_PASS           CONFIG_CSI_WIFI_PASSWORD
#else
#define CSI_WIFI_PASS           "prjmonitoramento2026"
#endif

#if CSI_USE_KCONFIG && defined(CONFIG_CSI_SERVER_IP)
#define CSI_SERVER_IP           CONFIG_CSI_SERVER_IP
#else
#define CSI_SERVER_IP           "192.168.1.140"
#endif

#if CSI_USE_KCONFIG && defined(CONFIG_CSI_SERVER_PORT)
#define CSI_SERVER_UDP_PORT     CONFIG_CSI_SERVER_PORT
#else
#define CSI_SERVER_UDP_PORT     5555
#endif

/* ----------------------------------------------------------------------------
 * 2. Taxa de Tráfego Ativo e Estímulo de Canal (50 Hz a 100 Hz)
 * ---------------------------------------------------------------------------- */
#if CSI_USE_KCONFIG && defined(CONFIG_CSI_TRAFFIC_HZ)
#define CSI_TRAFFIC_GEN_HZ      CONFIG_CSI_TRAFFIC_HZ
#else
#define CSI_TRAFFIC_GEN_HZ      60              /* 60 Hz garante amostragem CSI densa (<17 ms) */
#endif

#define CSI_MAX_TX_RATE_HZ      60              /* Limite máximo de envio UDP por nó */
#define CSI_PROBE_PORT          33333           /* Porta UDP descartável no gateway para probe */
#define CSI_LLTF_BYTES          128             /* 64 subportadoras HT20 * 2 bytes (I/Q int8_t) */
#define CSI_MIN_VALID_SIG_LEN   24              /* Descarta ACKs soltos (14 B) e quadros truncados */

/* ----------------------------------------------------------------------------
 * 3. Filtro de BSSID / Pares ESP32-S3 Autorizados (Anti-Fantasmas na Borda)
 *    Se CSI_ENFORCE_STATIC_BSSID = 1, trava no MAC fixo abaixo; caso contrário,
 *    trava automaticamente no BSSID do Cisco Linksys E1200 ao conectar.
 * ---------------------------------------------------------------------------- */
#define CSI_ENFORCE_STATIC_BSSID    0
static const uint8_t CSI_STATIC_TARGET_BSSID[6] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00};

/* Lista opcional de MACs de outros nós ESP32-S3 para malha multi-nó direta */
#define CSI_NUM_PEER_MACS           0
static const uint8_t CSI_PEER_MAC_LIST[3][6] = {
    {0x00, 0x00, 0x00, 0x00, 0x00, 0x00},
    {0x00, 0x00, 0x00, 0x00, 0x00, 0x00},
    {0x00, 0x00, 0x00, 0x00, 0x00, 0x00}
};

/* ----------------------------------------------------------------------------
 * 4. Protocolo Binário Compacto UDP (Little-Endian, __attribute__((packed)))
 *    Tamanho fixo do cabeçalho: 23 bytes + csi_len bytes de payload I/Q
 * ---------------------------------------------------------------------------- */
#define CSI_PACKET_MAGIC        0xC510
#define CSI_PACKET_VERSION      0x01

typedef struct __attribute__((packed)) {
    uint16_t magic;                             /* 0xC510 (2 bytes) */
    uint8_t  version;                           /* 0x01   (1 byte)  */
    uint8_t  node_id;                           /* 1..N   (1 byte)  */
    uint8_t  src_mac[6];                        /* BSSID/MAC transmissor (6 bytes) */
    uint64_t timestamp_us;                      /* Timestamp monotônico do chip em us (8 bytes) */
    int8_t   rssi;                              /* RSSI em dBm (1 byte signed) */
    int8_t   noise_floor;                       /* Piso de ruído RF em dBm (1 byte signed) */
    uint8_t  channel;                           /* Canal primário 2.4 GHz: 1, 6 ou 11 (1 byte) */
    uint16_t csi_len;                           /* Comprimento em bytes do vetor I/Q (2 bytes) */
    int8_t   csi_data[CSI_LLTF_BYTES];          /* 128 bytes: 64 pares [Imag, Real] int8_t */
} csi_udp_packet_t;

#endif /* CSI_CONFIG_H */
