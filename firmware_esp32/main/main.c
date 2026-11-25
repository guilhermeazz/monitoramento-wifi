/**
 * ============================================================================
 * main.c — Firmware ESP32-S3 para Sensoriamento Espacial Wi-Fi CSI
 * Compatível com ESP-IDF v5.x e v6.x
 * ============================================================================
 * Recursos de Engenharia Implementados:
 * 1. Rádio 802.11n HT20 Fixo (20 MHz) + Power Save Desabilitado (WIFI_PS_NONE).
 * 2. Filtragem na Borda (Anti-Fantasmas):
 *    - Whitelist estrita de MAC (BSSID do Cisco Linksys E1200 + Pares ESP32-S3).
 *    - Descarte de frames de controle curtos / ACKs soltos (sig_len < 24).
 *    - Descarte de quadros retransmitidos (802.11 FC Retry Bit) e burst duplicado.
 * 3. Metadados Físicos Completos: rssi, noise_floor, channel e timestamp_us.
 * 4. Gerador de Tráfego Ativo Controlado (50-100 Hz, padrão 60 Hz) para o gateway.
 * 5. Protocolo UDP Binário Compacto (csi_udp_packet_t, magic 0xC510, 151 bytes).
 * ============================================================================
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/event_groups.h"
#include "nvs_flash.h"
#include "esp_system.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_timer.h"
#include "esp_random.h"
#include "lwip/err.h"
#include "lwip/sockets.h"
#include "lwip/sys.h"
#include "lwip/netdb.h"
#include "lwip/inet.h"

#include "csi_config.h"

static const char *TAG = "CSI_S3_NODE";

static EventGroupHandle_t s_wifi_event_group;
#define WIFI_CONNECTED_BIT BIT0

static int s_udp_sock = -1;
static struct sockaddr_in s_server_addr;
static esp_ip4_addr_t s_gateway_ip;
static int64_t s_last_csi_tx_us = 0;

/* BSSID do Roteador Cisco Linksys E1200 autenticado */
static uint8_t s_ap_bssid[6] = {0};
static bool s_has_ap_bssid = false;

/* Deduplicador leve de rajadas de retransmissão (< 1.5 ms com mesmo RSSI/len) */
static int64_t s_prev_rx_us = 0;
static uint32_t s_prev_sig_len = 0;
static int8_t s_prev_rssi = 0;

/* ============================================================================
 * Validação de MAC Autorizado (Roteador E1200 ou Pares ESP32-S3)
 * ============================================================================ */
static inline bool is_authorized_source_mac(const uint8_t *mac)
{
#if CSI_ENFORCE_STATIC_BSSID
    if (memcmp(mac, CSI_STATIC_TARGET_BSSID, 6) == 0) {
        return true;
    }
#else
    if (s_has_ap_bssid && memcmp(mac, s_ap_bssid, 6) == 0) {
        return true;
    }
#endif

    for (int i = 0; i < CSI_NUM_PEER_MACS; i++) {
        if (memcmp(mac, CSI_PEER_MAC_LIST[i], 6) == 0) {
            return true;
        }
    }
    return false;
}

/* ============================================================================
 * Callback de Recepção de Wi-Fi CSI (Execução Rápida Zero-Alloc)
 * ============================================================================ */
static void IRAM_ATTR wifi_csi_rx_cb(void *ctx, wifi_csi_info_t *info)
{
    if (!info || !info->buf || s_udp_sock < 0) {
        return;
    }

    /* 1. Exige vetor HT20 LLTF íntegro (128 bytes = 64 subportadoras I/Q) sem canal secundário */
    if (info->len < CSI_LLTF_BYTES || info->first_word_invalid || info->rx_ctrl.secondary_channel != 0) {
        return;
    }

    /* 2. Descarta ACKs soltos e quadros de controle curtos que geram picos espúrios */
    if (info->rx_ctrl.sig_len < CSI_MIN_VALID_SIG_LEN) {
        return;
    }

    /* 3. Filtra apenas pacotes vindos do BSSID autorizado (Cisco Linksys E1200 ou par ESP32-S3) */
    if (!is_authorized_source_mac(info->mac)) {
        return;
    }

    /* 4. Descarta frames com flag de retransmissão 802.11 (Frame Control Byte 1, bit 3 = Retry) */
    if (info->hdr != NULL) {
        const uint8_t *fc = (const uint8_t *)info->hdr;
        if ((fc[1] & 0x08) != 0) {
            return; /* Frame retransmitido (retry == 1) descartado */
        }
    }

    int64_t now_us = esp_timer_get_time();

    /* Proteção adicional contra rajadas de retransmissão sub-milissegundo */
    if ((now_us - s_prev_rx_us) < 1500 &&
        info->rx_ctrl.sig_len == s_prev_sig_len &&
        info->rx_ctrl.rssi == s_prev_rssi) {
        return;
    }
    s_prev_rx_us = now_us;
    s_prev_sig_len = info->rx_ctrl.sig_len;
    s_prev_rssi = (int8_t)info->rx_ctrl.rssi;

    /* 5. Controle de taxa máxima de transmissão UDP (ex: 60 Hz) */
    if ((now_us - s_last_csi_tx_us) < (1000000LL / CSI_MAX_TX_RATE_HZ)) {
        return;
    }
    s_last_csi_tx_us = now_us;

    /* 6. Empacotamento Binário Little-Endian (__attribute__((packed))) */
    csi_udp_packet_t pkt;
    pkt.magic        = CSI_PACKET_MAGIC;
    pkt.version      = CSI_PACKET_VERSION;
    pkt.node_id      = (uint8_t)CSI_NODE_ID_NUM;
    memcpy(pkt.src_mac, info->mac, 6);
    pkt.timestamp_us = (uint64_t)now_us;
    pkt.rssi         = (int8_t)info->rx_ctrl.rssi;
    pkt.noise_floor  = (int8_t)info->rx_ctrl.noise_floor;
    pkt.channel      = (uint8_t)info->rx_ctrl.channel;
    pkt.csi_len      = (uint16_t)CSI_LLTF_BYTES;
    memcpy(pkt.csi_data, info->buf, CSI_LLTF_BYTES);

    sendto(
        s_udp_sock,
        &pkt,
        sizeof(csi_udp_packet_t),
        MSG_DONTWAIT,
        (struct sockaddr *)&s_server_addr,
        sizeof(s_server_addr)
    );
}

/* ============================================================================
 * Inicialização do Subsistema CSI no ESP32-S3
 * ============================================================================ */
static void init_wifi_csi(void)
{
    wifi_csi_config_t csi_config = {
        .lltf_en           = true,
        .htltf_en          = true,
        .stbc_htltf2_en    = false,
        .ltf_merge_en      = true,
        .channel_filter_en = true,
        .manu_scale        = false,
        .shift             = false,
    };

    ESP_ERROR_CHECK(esp_wifi_set_csi_config(&csi_config));
    ESP_ERROR_CHECK(esp_wifi_set_csi_rx_cb(&wifi_csi_rx_cb, NULL));
    ESP_ERROR_CHECK(esp_wifi_set_csi(true));

    ESP_LOGI(
        TAG,
        "CSI Binário (0xC510) ativo: NODE_ID=%d | HT20 128B | MaxRate=%d Hz",
        CSI_NODE_ID_NUM,
        CSI_MAX_TX_RATE_HZ
    );
}

/* ============================================================================
 * Task Geradora de Tráfego Ativo Controlado (50–100 Hz) para o Roteador/Servidor
 * ============================================================================ */
static void traffic_generator_task(void *pvParameters)
{
    struct sockaddr_in gw_addr;
    memset(&gw_addr, 0, sizeof(gw_addr));
    gw_addr.sin_family = AF_INET;
    gw_addr.sin_port = htons(CSI_PROBE_PORT);
    gw_addr.sin_addr.s_addr = s_gateway_ip.addr;

    const uint8_t probe_payload[8] = {0xC5, 0x10, (uint8_t)CSI_NODE_ID_NUM, 0x00, 0x50, 0x49, 0x4E, 0x47};
    const uint32_t base_period_ms = 1000 / CSI_TRAFFIC_GEN_HZ;

    ESP_LOGI(TAG, "Tráfego ativo de sondagem CSI iniciado (%d Hz, período ~%lu ms).",
             CSI_TRAFFIC_GEN_HZ, (unsigned long)base_period_ms);

    while (1) {
        if (s_udp_sock >= 0 && s_gateway_ip.addr != 0) {
            sendto(
                s_udp_sock,
                probe_payload,
                sizeof(probe_payload),
                MSG_DONTWAIT,
                (struct sockaddr *)&gw_addr,
                sizeof(gw_addr)
            );
        }
        /* Micro-jitter (0..2 ms) para evitar sincronização entre múltiplos nós ESP32-S3 */
        uint32_t jitter_ms = esp_random() % 3;
        vTaskDelay(pdMS_TO_TICKS(base_period_ms + jitter_ms));
    }
}

/* ============================================================================
 * Handler de Eventos Wi-Fi e IP
 * ============================================================================ */
static void wifi_event_handler(void *arg, esp_event_base_t event_base, int32_t event_id, void *event_data)
{
    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_CONNECTED) {
        wifi_event_sta_connected_t *conn = (wifi_event_sta_connected_t *)event_data;
        memcpy(s_ap_bssid, conn->bssid, 6);
        s_has_ap_bssid = true;
        ESP_LOGI(
            TAG,
            "Associado ao AP Cisco Linksys E1200 | BSSID: %02x:%02x:%02x:%02x:%02x:%02x | Canal: %d",
            s_ap_bssid[0], s_ap_bssid[1], s_ap_bssid[2],
            s_ap_bssid[3], s_ap_bssid[4], s_ap_bssid[5],
            conn->channel
        );
    } else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED) {
        s_has_ap_bssid = false;
        ESP_LOGW(TAG, "Conexão Wi-Fi perdida. Tentando reconectar...");
        esp_wifi_connect();
        xEventGroupClearBits(s_wifi_event_group, WIFI_CONNECTED_BIT);
    } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t *event = (ip_event_got_ip_t *)event_data;
        s_gateway_ip = event->ip_info.gw;
        ESP_LOGI(
            TAG,
            "IP Obtido: " IPSTR " | Gateway AP: " IPSTR " | Servidor Destino: %s:%d",
            IP2STR(&event->ip_info.ip),
            IP2STR(&s_gateway_ip),
            CSI_SERVER_IP,
            CSI_SERVER_UDP_PORT
        );
        xEventGroupSetBits(s_wifi_event_group, WIFI_CONNECTED_BIT);
    }
}

/* ============================================================================
 * Inicialização do Rádio Wi-Fi (802.11n HT20 Estável + WIFI_PS_NONE)
 * ============================================================================ */
static void wifi_init_sta(void)
{
    s_wifi_event_group = xEventGroupCreate();

    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    esp_event_handler_instance_t instance_any_id;
    esp_event_handler_instance_t instance_got_ip;
    ESP_ERROR_CHECK(esp_event_handler_instance_register(
        WIFI_EVENT, ESP_EVENT_ANY_ID, &wifi_event_handler, NULL, &instance_any_id));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(
        IP_EVENT, IP_EVENT_STA_GOT_IP, &wifi_event_handler, NULL, &instance_got_ip));

    wifi_config_t wifi_config = {
        .sta = {
            .ssid = CSI_WIFI_SSID,
            .password = CSI_WIFI_PASS,
            .threshold.authmode = WIFI_AUTH_WPA2_PSK,
        },
    };

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));

    /* Força protocolo 802.11b/g/n e largura de banda estrita de 20 MHz (HT20) */
    ESP_ERROR_CHECK(esp_wifi_set_protocol(WIFI_IF_STA, WIFI_PROTOCOL_11B | WIFI_PROTOCOL_11G | WIFI_PROTOCOL_11N));
    ESP_ERROR_CHECK(esp_wifi_set_bandwidth(WIFI_IF_STA, WIFI_BW20));
    ESP_ERROR_CHECK(esp_wifi_start());

    /* Desabilita completamente economia de energia do modem Wi-Fi */
    ESP_ERROR_CHECK(esp_wifi_set_ps(WIFI_PS_NONE));
}

void app_main(void)
{
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    ESP_LOGI(TAG, "Iniciando Sensor ESP32-S3 Wi-Fi CSI [node_id=%d]...", CSI_NODE_ID_NUM);
    wifi_init_sta();

    xEventGroupWaitBits(s_wifi_event_group, WIFI_CONNECTED_BIT, pdFALSE, pdTRUE, portMAX_DELAY);
    ESP_ERROR_CHECK(esp_wifi_set_ps(WIFI_PS_NONE));

    s_udp_sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_IP);
    if (s_udp_sock < 0) {
        ESP_LOGE(TAG, "Erro fatal ao criar socket UDP!");
        return;
    }

    memset(&s_server_addr, 0, sizeof(s_server_addr));
    s_server_addr.sin_family = AF_INET;
    s_server_addr.sin_port = htons(CSI_SERVER_UDP_PORT);
    inet_pton(AF_INET, CSI_SERVER_IP, &s_server_addr.sin_addr.s_addr);

    init_wifi_csi();
    xTaskCreatePinnedToCore(traffic_generator_task, "csi_traffic_gen", 4096, NULL, 5, NULL, 0);
}
