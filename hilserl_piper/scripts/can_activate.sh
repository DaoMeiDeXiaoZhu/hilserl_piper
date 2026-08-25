#!/usr/bin/env bash
# 激活 Piper USB-CAN（socketcan，1Mbps）。
#
# 用法（在 ~/桌面/lerobot_hilserl 下）:
#   bash scripts/can_activate.sh                 # 读 hardware.json：从臂 + 主臂（若有）
#   bash scripts/can_activate.sh can0
#   bash scripts/can_activate.sh can0 1000000
#   bash scripts/can_activate.sh can0 1000000 1-2:1.0   # 多 CAN 时指定 USB bus-info
#
# 无参数时：control.follower_can，以及 leader_follower 时的 leader_can。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HW_JSON="${SCRIPT_DIR}/../cfg/hardware.json"

list_cfg_cans() {
  if [[ -f "$HW_JSON" ]] && command -v python3 >/dev/null 2>&1; then
    python3 - "$HW_JSON" <<'PY'
import json, sys
p = sys.argv[1]
try:
    d = json.load(open(p, encoding="utf-8"))
except Exception:
    print("can0")
    raise SystemExit(0)
ctrl = d.get("control") or {}
can = ctrl.get("can") or ctrl.get("follower_can") or d.get("can_port") or "can0"
print(can or "can0")
# 固件主从 / 同口：只激活一条 CAN
if ctrl.get("firmware_ms") or (ctrl.get("leader_can") in (None, "", can)):
    raise SystemExit(0)
leader = ctrl.get("leader_can")
mode = ctrl.get("mode") or ""
if mode == "leader_follower" and leader and leader != can:
    print(leader)
PY
    return
  fi
  echo "can0"
}

need_pkg() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "缺少 $1，请安装: sudo apt update && sudo apt install ethtool can-utils"
    exit 1
  fi
}

need_pkg ip
need_pkg ethtool

if ! lsmod | grep -q '^gs_usb'; then
  sudo modprobe gs_usb || true
fi

bus_info() {
  sudo ethtool -i "$1" 2>/dev/null | awk '/bus-info/ {print $2}'
}

refresh_ifaces() {
  mapfile -t IFACES < <(ip -br link show type can 2>/dev/null | awk '{print $1}' || true)
  COUNT="${#IFACES[@]}"
  if [[ "$COUNT" -eq 1 && -z "${IFACES[0]:-}" ]]; then
    COUNT=0
  fi
}

list_ifaces() {
  local iface bus
  for iface in "${IFACES[@]}"; do
    [[ -z "$iface" ]] && continue
    bus="$(bus_info "$iface")"
    echo "  ${iface}  usb=${bus:-unknown}  $(ip -br link show "$iface" | awk '{print $2}')"
  done
}

activate_one() {
  local CAN_NAME="$1"
  local BITRATE="$2"
  local USB_ADDRESS="${3:-}"

  refresh_ifaces
  echo "==================== CAN activate ===================="
  echo "  name=${CAN_NAME}  bitrate=${BITRATE}${USB_ADDRESS:+  usb=${USB_ADDRESS}}"

  if [[ "$COUNT" -eq 0 ]]; then
    echo "未检测到 CAN 网卡。请插上 Piper USB-CAN 后再试。"
    echo "若刚插入: sudo modprobe gs_usb"
    return 1
  fi

  local IFACE=""
  if [[ -n "$USB_ADDRESS" ]]; then
    local iface
    for iface in "${IFACES[@]}"; do
      if [[ "$(bus_info "$iface")" == "$USB_ADDRESS" ]]; then
        IFACE="$iface"
        break
      fi
    done
    if [[ -z "$IFACE" ]]; then
      echo "找不到 USB=${USB_ADDRESS} 对应的 CAN 口。当前:"
      list_ifaces
      return 1
    fi
  elif ip link show "$CAN_NAME" >/dev/null 2>&1; then
    IFACE="$CAN_NAME"
  else
    echo "没有接口 ${CAN_NAME}。请先插上对应 USB-CAN，或带 bus-info："
    echo "  bash scripts/can_activate.sh ${CAN_NAME} ${BITRATE} 1-2:1.0"
    echo "当前:"
    list_ifaces
    return 1
  fi

  echo "  选用接口 ${IFACE}  usb=$(bus_info "$IFACE")"

  local IS_UP="no"
  if ip link show "$IFACE" | grep -q "UP"; then
    IS_UP="yes"
  fi
  local CUR_BR
  CUR_BR="$(ip -details link show "$IFACE" | grep -oP 'bitrate \K\d+' || true)"

  sudo ip link set "$IFACE" down
  sudo ip link set "$IFACE" type can bitrate "$BITRATE"
  if [[ "$IFACE" != "$CAN_NAME" ]]; then
    echo "  重命名 ${IFACE} -> ${CAN_NAME}"
    sudo ip link set "$IFACE" name "$CAN_NAME"
    IFACE="$CAN_NAME"
  fi
  sudo ip link set "$IFACE" up

  echo "  ${IFACE} UP  bitrate=${BITRATE}  (was up=${IS_UP} bitrate=${CUR_BR:-none})"
  ip -br link show "$IFACE"
  echo "==================== done ============================"
}

BITRATE_DEFAULT=1000000

if [[ $# -eq 0 ]]; then
  mapfile -t TARGETS < <(list_cfg_cans)
  if [[ "${#TARGETS[@]}" -eq 0 ]]; then
    TARGETS=(can0)
  fi
  echo "将激活: ${TARGETS[*]}"
  for name in "${TARGETS[@]}"; do
    [[ -z "$name" ]] && continue
    activate_one "$name" "$BITRATE_DEFAULT" ""
  done
else
  activate_one "${1}" "${2:-$BITRATE_DEFAULT}" "${3:-}"
fi
