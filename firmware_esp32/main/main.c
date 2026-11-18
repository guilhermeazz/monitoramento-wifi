/**
 * ============================================================================
 * Firmware ESP32 / ESP32-S3 — Sensor de Monitoramento Espacial Wi-Fi CSI
 * Compatível com ESP-IDF v5.x e v6.x (Blindagem de Sinal & Filtro BSSID)
 * ============================================================================
 * Pré-requisito no sdkconfig (sdkconfig.defaults):
 *   CONFIG_ESP_WIFI_CSI_ENABLED=y
 *
 * Blindagens Implementadas:
 * 1. Power Save Desativado (WIFI_PS_NONE) antes e logo após esp_wifi_start()
 *    para impedir micro-sleep do modem e eliminar buracos de amostragem.
 * 2. Filtro de BSSID (MAC do Roteador AP conectado) no callback wifi_csi_rx_cb,
 *    ignorando probe requests e pacotes espúrios de celulares vizinhos.
 * 3. Limpeza de Subportadoras Inválidas (Guard Bands & DC): em buffers HT20 de
 *    128 bytes (64 subportadoras), descarta as primeiras 6 (0..5), as últimas 5
 *    (59..63) e a subportadora DC central (32), enviando apenas as 52
 *    subportadoras LLTF/HT-LTF válidas (104 inteiros I/Q) a 30 Hz via UDP.
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

/* ============================================================================
 * MACROS DE CONFIGURAÇÃO RÁPIDA (MANTIDAS COM AS CREDENCIAIS DO PROJETO)
 * ============================================================================ */
#define NODE_ID             "esp_3"                /* Identificador: "esp_1", "esp_2" ou "esp_3" */
#define WIFI_SSID           "CSI_RADAR_2G"         /* SSID do Roteador Wi-Fi 2.4 GHz */
#define WIFI_PASS           "prjmonitoramento2026" /* Senha do Roteador Wi-Fi */
#define SERVER_IP           "192.168.1.132"        /* IP do Notebook rodando o Backend FastAPI */
#define SERVER_UDP_PORT     5555                   /* Porta UDP do Receptor CSI no Backend */

#define TRAFFIC_GEN_HZ      15                     /* Estímulo de pacotes UDP ao Roteador (15 Hz = ~66ms + jitter) */
#define CSI_MAX_SEND_HZ     20                     /* Envio estável de CSI a 20 Hz por ESP32 (evita colisão entre 3 ESPs) */
#define MAX_CSI_BYTES       128                    /* 64 subportadoras HT20 * 2 bytes (Imag, Real) */

static const char *TAG = "CSI_NODE";

static EventGroupHandle_t s_wifi_event_group;
#define WIFI_CONNECTED_BIT BIT0

static int s_udp_sock = -1;
static struct sockaddr_in s_server_addr;
static esp_ip4_addr_t s_gateway_ip;
static int64_t s_last_csi_tx_us = 0;

/* Armazena o BSSID (MAC) do Roteador AP conectado para filtrar pacotes de terceiros */
static uint8_t s_ap_bssid[6] = {0};
static bool s_has_ap_bssid = false;

/* ============================================================================
 * Callback de Recepção de Wi-Fi CSI (Filtro BSSID + Limpeza de 52 Subportadoras)
 * ============================================================================ */
static void wifi_csi_rx_cb(void *ctx, wifi_csi_info_t *info)
{
    /* 1. Validação estrita de integridade física:
     * - info e buffer válidos
     * - len EXATAMENTE 128 bytes (64 pares I/Q de 8 bits do canal HT20)
     * - descarta pacotes com first_word_invalid ou canal secundário (evita saltos de subportadoras)
     */
    if (!info || !info->buf || info->len != 128 || info->first_word_invalid || s_udp_sock < 0) {
        return;
    }
    if (info->rx_ctrl.secondary_channel != 0) {
        return;
    }

    /* 2. Filtro estrito de BSSID: aceita apenas quadros vindos do Roteador AP conectado */
    if (s_has_ap_bssid && memcmp(info->mac, s_ap_bssid, 6) != 0) {
        return;
    }

    /* 3. Controle de taxa de amostragem para 30 Hz constantes */
    int64_t now_us = esp_timer_get_time();
    if ((now_us - s_last_csi_tx_us) < (1000000LL / CSI_MAX_SEND_HZ)) {
        return;
    }
    s_last_csi_tx_us = now_us;

    int rssi = info->rx_ctrl.rssi;
    const int8_t *csi_buf = info->buf;

    char json_payload[1024];
    int offset = snprintf(
        json_payload,
        sizeof(json_payload),
        "{\"node_id\":\"%s\",\"rssi\":%d,\"csi\":[",
        NODE_ID,
        rssi
    );

    /* 4. Extração exata das 52 subportadoras válidas do LLTF/HT-LTF (int8_t signed):
     * - Descarta índices de guarda inferiores: 0..5
     * - Descarta subportadora DC nula central: índice 32
     * - Descarta índices de guarda superiores: 59..63
     * Coletando estritamente os 52 pares válidos: 6..31 (26 pares) e 33..58 (26 pares).
     */
    bool first = true;
    for (int sc = 6; sc <= 58; sc++) {
        if (sc == 32) {
            continue; /* Pula subportadora DC central */
        }
        int imag_val = (int)csi_buf[sc * 2];
        int real_val = (int)csi_buf[sc * 2 + 1];
        int written = snprintf(
            json_payload + offset,
            sizeof(json_payload) - offset,
            first ? "%d,%d" : ",%d,%d",
            imag_val,
            real_val
        );
        first = false;
        if (written < 0 || (offset + written) >= (int)sizeof(json_payload) - 4) {
            break;
        }
        offset += written;
    }

    snprintf(json_payload + offset, sizeof(json_payload) - offset, "]}");

    /* Envia datagrama UDP não-bloqueante para o Notebook Servidor na porta 5555 */
    sendto(
        s_udp_sock,
        json_payload,
        strlen(json_payload),
        MSG_DONTWAIT,
        (struct sockaddr *)&s_server_addr,
        sizeof(s_server_addr)
    );
}

/* ============================================================================
 * Configuração do Subsistema Wi-Fi CSI no ESP32 / ESP32-S3
 * ============================================================================ */
static void init_wifi_csi(void)
{
    wifi_csi_config_t csi_config = {
        .lltf_en           = true,
        .htltf_en          = true,
        .stbc_htltf2_en    = true,
        .ltf_merge_en      = true,
        .channel_filter_en = true,
        .manu_scale        = false,
        .shift             = false,
    };

    ESP_ERROR_CHECK(esp_wifi_set_csi_config(&csi_config));
    ESP_ERROR_CHECK(esp_wifi_set_csi_rx_cb(&wifi_csi_rx_cb, NULL));
    ESP_ERROR_CHECK(esp_wifi_set_csi(true));

    ESP_LOGI(TAG, "Captura de Wi-Fi CSI (52 subportadoras limpas @ %d Hz) ativa para NODE_ID=%s",
             CSI_MAX_SEND_HZ, NODE_ID);
}

/* ============================================================================
 * Task FreeRTOS Geradora de Tráfego UDP Contínuo para o Roteador (35 Hz)
 * ============================================================================ */
static void traffic_generator_task(void *pvParameters)
{
    struct sockaddr_in gw_addr;
    memset(&gw_addr, 0, sizeof(gw_addr));
    gw_addr.sin_family = AF_INET;
    gw_addr.sin_port = htons(33333);
    gw_addr.sin_addr.s_addr = s_gateway_ip.addr;

    const char *probe_pkt = "CSI_PROBE";
    const uint32_t base_delay_ms = 1000 / TRAFFIC_GEN_HZ; /* 66 ms (15 Hz) */

    ESP_LOGI(TAG, "Task geradora de tráfego RF iniciada (%d Hz + jitter anti-colisão 0..9ms).", TRAFFIC_GEN_HZ);

    while (1) {
        if (s_udp_sock >= 0 && s_gateway_ip.addr != 0) {
            sendto(
                s_udp_sock,
                probe_pkt,
                strlen(probe_pkt),
                MSG_DONTWAIT,
                (struct sockaddr *)&gw_addr,
                sizeof(gw_addr)
            );
        }
        /* Jitter aleatório (0 a 9 ms) para evitar colisão de pacotes entre múltiplos ESP32-S3 */
        uint32_t jitter_ms = esp_random() % 10;
        vTaskDelay(pdMS_TO_TICKS(base_delay_ms + jitter_ms));
    }
}

/* ============================================================================
 * Handler de Eventos Wi-Fi e IP (Registra o BSSID do Roteador AP)
 * ============================================================================ */
static void wifi_event_handler(void *arg, esp_event_base_t event_base, int32_t event_id, void *event_data)
{
    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_CONNECTED) {
        wifi_event_sta_connected_t *conn = (wifi_event_sta_connected_t *)event_data;
        memcpy(s_ap_bssid, conn->bssid, 6);
        s_has_ap_bssid = true;
        ESP_LOGI(TAG, "Filtro BSSID travado no Roteador AP: %02x:%02x:%02x:%02x:%02x:%02x",
                 s_ap_bssid[0], s_ap_bssid[1], s_ap_bssid[2],
                 s_ap_bssid[3], s_ap_bssid[4], s_ap_bssid[5]);
    } else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED) {
        s_has_ap_bssid = false;
        ESP_LOGW(TAG, "Wi-Fi desconectado. Reconectando ao roteador 2.4 GHz...");
        esp_wifi_connect();
        xEventGroupClearBits(s_wifi_event_group, WIFI_CONNECTED_BIT);
    } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t *event = (ip_event_got_ip_t *)event_data;
        s_gateway_ip = event->ip_info.gw;
        ESP_LOGI(TAG, "Conectado! IP local: " IPSTR " | Gateway TX: " IPSTR,
                 IP2STR(&event->ip_info.ip), IP2STR(&s_gateway_ip));
        xEventGroupSetBits(s_wifi_event_group, WIFI_CONNECTED_BIT);
    }
}

/* ============================================================================
 * Inicialização da Estação Wi-Fi (802.11b/g/n 2.4 GHz HT20 + WIFI_PS_NONE)
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
            .ssid = WIFI_SSID,
            .password = WIFI_PASS,
            .threshold.authmode = WIFI_AUTH_WPA_WPA2_PSK,
        },
    };

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));
    ESP_ERROR_CHECK(esp_wifi_set_bandwidth(WIFI_IF_STA, WIFI_BW20));
    ESP_ERROR_CHECK(esp_wifi_start());

    /* Garante desativação total do Power Save (WIFI_PS_NONE) logo após iniciar o Wi-Fi */
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

    ESP_LOGI(TAG, "Inicializando Nó Sensor Wi-Fi CSI [%s]...", NODE_ID);
    wifi_init_sta();

    /* Aguarda conexão com o Roteador Wi-Fi */
    xEventGroupWaitBits(s_wifi_event_group, WIFI_CONNECTED_BIT, pdFALSE, pdTRUE, portMAX_DELAY);

    /* Reforça WIFI_PS_NONE após conectar ao AP */
    esp_wifi_set_ps(WIFI_PS_NONE);

    /* Cria socket UDP para envio dos pacotes CSI ao Notebook Servidor */
    s_udp_sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_IP);
    if (s_udp_sock < 0) {
        ESP_LOGE(TAG, "Falha ao criar socket UDP!");
        return;
    }

    memset(&s_server_addr, 0, sizeof(s_server_addr));
    s_server_addr.sin_family = AF_INET;
    s_server_addr.sin_port = htons(SERVER_UDP_PORT);
    inet_pton(AF_INET, SERVER_IP, &s_server_addr.sin_addr.s_addr);

    /* Ativa captura de CSI e inicia task geradora de tráfego */
    init_wifi_csi();
    xTaskCreate(traffic_generator_task, "traffic_gen_task", 4096, NULL, 5, NULL);
}
