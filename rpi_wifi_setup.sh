#!/bin/bash

set -e

SSID="DRON_WIFI"
PASS="12345678"
MAC="88:a2:9e:67:f9:68"
IP="192.168.50.10"

apt update
apt install -y network-manager dnsmasq-base

systemctl enable --now NetworkManager
# systemctl start NetworkManager

mkdir -p /etc/NetworkManager/dnsmasq-shared.d

cat >/etc/NetworkManager/dnsmasq-shared.d/hotspot.conf <<EOF
dhcp-host=$MAC,$IP,infinite
dhcp-range=192.168.50.20,192.168.50.22,255.255.255.0,12h
EOF

nmcli connection delete Hotspot 2>/dev/null || true

nmcli connection add type wifi ifname wlan0 con-name Hotspot autoconnect yes ssid "$SSID"
nmcli connection modify Hotspot wifi.mode ap wifi.band bg wifi-sec.key-mgmt wpa-psk wifi-sec.psk "$PASS" ipv4.method shared ipv4.addresses 192.168.50.1/24 ipv6.method disabled connection.autoconnect yes
nmcli connection up Hotspot
