cmake_minimum_required(VERSION 3.16)

# Habilita suporte a Wi-Fi CSI no menuconfig / sdkconfig caso necessário
include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(wifi_csi_spatial_node)
