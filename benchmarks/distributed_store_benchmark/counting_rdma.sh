#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 [window_seconds] [rdma_device ...]" >&2
  echo "       DEVS=mlx5_0,mlx5_1 $0 [window_seconds]" >&2
  echo "       PORT=1 COUNTERS=port_rcv_data,port_xmit_data $0 [window_seconds] [rdma_device ...]" >&2
  echo "       NETDEVS=fabric0,fabric1 $0 [window_seconds]   # optional Linux netdev counters" >&2
  echo "       PORT defaults to all RDMA ports; set PORT=1 to restrict it." >&2
  echo "       RDMA port counters are used for both InfiniBand and RoCE." >&2
}

WINDOW=${WINDOW:-5}
if (($# > 0)); then
  case "$1" in
    -h | --help)
      usage
      exit 0
      ;;
  esac

  if [[ "$1" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    WINDOW=$1
    shift
  fi
fi

if [[ ! "$WINDOW" =~ ^[0-9]+([.][0-9]+)?$ ]] ||
  ! awk -v window="$WINDOW" 'BEGIN { exit !(window > 0) }'; then
  echo "window_seconds must be a positive number: $WINDOW" >&2
  usage
  exit 2
fi

if [[ -n "${MODE:-}" ]]; then
  echo "Ignoring MODE=$MODE; RDMA port counters are used for both InfiniBand and RoCE." >&2
fi

PORT=${PORT:-}
COUNTER=${COUNTER:-}
COUNTERS=${COUNTERS:-}
NET_COUNTERS=${NET_COUNTERS:-rx_bytes,tx_bytes}
SYSFS_INFINIBAND=${SYSFS_INFINIBAND:-/sys/class/infiniband}
SYSFS_NET=${SYSFS_NET:-/sys/class/net}
USER_DEVS=${DEVS:-}
USER_NETDEVS=${NETDEVS:-}
RDMA_COUNTERS=()
NETDEV_COUNTERS=()

if [[ -n "$PORT" && ! "$PORT" =~ ^[0-9]+$ ]]; then
  echo "PORT must be a non-negative integer: $PORT" >&2
  usage
  exit 2
fi

if [[ -n "$COUNTER" ]]; then
  RDMA_COUNTERS=("$COUNTER")
elif [[ -n "$COUNTERS" ]]; then
  read -r -a RDMA_COUNTERS <<< "${COUNTERS//,/ }"
else
  RDMA_COUNTERS=(port_rcv_data port_xmit_data)
fi

read -r -a NETDEV_COUNTERS <<< "${NET_COUNTERS//,/ }"

DEVS=()
if (($# > 0)); then
  DEVS=("$@")
elif [[ -n "$USER_DEVS" ]]; then
  read -r -a DEVS <<< "${USER_DEVS//,/ }"
elif [[ -d "$SYSFS_INFINIBAND" ]]; then
  for dev_path in "$SYSFS_INFINIBAND"/*; do
    [[ -e "$dev_path" ]] || continue
    DEVS+=("${dev_path##*/}")
  done
fi

NETDEVS_LIST=()
if [[ -n "$USER_NETDEVS" ]]; then
  read -r -a NETDEVS_LIST <<< "${USER_NETDEVS//,/ }"
fi

ACTIVE_LABELS=()
ACTIVE_RATE_NAMES=()
COUNTER_PATHS=()
BEFORE=()
MULTIPLIERS=()

rate_name_for_counter() {
  local counter=$1

  case "$counter" in
    port_rcv_data | rx_bytes)
      echo "rcv_rate"
      ;;
    port_xmit_data | tx_bytes)
      echo "xmit_rate"
      ;;
    *)
      echo "${counter}_rate"
      ;;
  esac
}

add_sample() {
  local label=$1
  local counter_path=$2
  local multiplier=$3
  local source=$4
  local rate_name=$5
  local before

  if [[ ! -r "$counter_path" ]]; then
    echo "Skipping $label: $source counter not found or not readable: $counter_path" >&2
    return 1
  fi

  before=$(<"$counter_path")
  if [[ ! "$before" =~ ^[0-9]+$ ]]; then
    echo "Skipping $label: $source counter value is not numeric: $counter_path" >&2
    return 1
  fi

  ACTIVE_LABELS+=("$label")
  ACTIVE_RATE_NAMES+=("$rate_name")
  COUNTER_PATHS+=("$counter_path")
  BEFORE+=("$before")
  MULTIPLIERS+=("$multiplier")
  return 0
}

add_rdma_samples() {
  local d=$1
  local added=1
  local port_dir
  local port
  local counter
  local label

  if [[ -n "$PORT" ]]; then
    for counter in "${RDMA_COUNTERS[@]}"; do
      label="$d"
      if ((${#RDMA_COUNTERS[@]} > 1)); then
        label="$d/$counter"
      fi
      add_sample "$label" "$SYSFS_INFINIBAND/$d/ports/$PORT/counters/$counter" 4 "RDMA" "$(rate_name_for_counter "$counter")" && added=0
    done
    return "$added"
  fi

  if [[ -d "$SYSFS_INFINIBAND/$d/ports" ]]; then
    for port_dir in "$SYSFS_INFINIBAND/$d/ports"/*; do
      [[ -d "$port_dir" ]] || continue
      port=${port_dir##*/}
      for counter in "${RDMA_COUNTERS[@]}"; do
        label="$d/port$port"
        if ((${#RDMA_COUNTERS[@]} > 1)); then
          label="$label/$counter"
        fi
        add_sample "$label" "$port_dir/counters/$counter" 4 "RDMA" "$(rate_name_for_counter "$counter")" && added=0
      done
    done
  fi

  return "$added"
}

add_netdev_samples() {
  local ndev=$1
  local counter

  for counter in "${NETDEV_COUNTERS[@]}"; do
    add_sample "$ndev/$counter" "$SYSFS_NET/$ndev/statistics/$counter" 1 "netdev" "$(rate_name_for_counter "$counter")" || true
  done
}

for d in "${DEVS[@]+"${DEVS[@]}"}"; do
  add_rdma_samples "$d" || true
done

if [[ -n "$USER_NETDEVS" ]]; then
  for ndev in "${NETDEVS_LIST[@]+"${NETDEVS_LIST[@]}"}"; do
    add_netdev_samples "$ndev"
  done
fi

if ((${#ACTIVE_LABELS[@]} == 0)); then
  echo "No readable counters found." >&2
  if [[ -n "$PORT" ]]; then
    echo "Searched RDMA counters under $SYSFS_INFINIBAND/<dev>/ports/$PORT/counters/{${RDMA_COUNTERS[*]}}" >&2
  else
    echo "Searched RDMA counters under $SYSFS_INFINIBAND/<dev>/ports/<port>/counters/{${RDMA_COUNTERS[*]}}" >&2
  fi
  if [[ -n "$USER_NETDEVS" ]]; then
    echo "Searched netdev counters under $SYSFS_NET/<netdev>/statistics/{${NETDEV_COUNTERS[*]}}" >&2
  fi
  if [[ -d "$SYSFS_INFINIBAND" ]]; then
    echo "Available RDMA devices under $SYSFS_INFINIBAND:" >&2
    for dev_path in "$SYSFS_INFINIBAND"/*; do
      [[ -e "$dev_path" ]] || continue
      echo "  ${dev_path##*/}" >&2
    done
  fi
  if [[ -d "$SYSFS_NET" ]]; then
    echo "Available netdev counters under $SYSFS_NET:" >&2
    for net_path in "$SYSFS_NET"/*; do
      [[ -r "$net_path/statistics/rx_bytes" ]] || continue
      echo "  ${net_path##*/}" >&2
    done
  fi
  usage
  exit 1
fi

while true; do
  sleep "$WINDOW"
  echo "=================================="
  echo "== $(date '+%Y-%m-%d %H:%M:%S') =="

  for i in "${!ACTIVE_LABELS[@]}"; do
    label=${ACTIVE_LABELS[$i]}
    rate_name=${ACTIVE_RATE_NAMES[$i]}
    counter_path=${COUNTER_PATHS[$i]}
    before=${BEFORE[$i]}
    multiplier=${MULTIPLIERS[$i]}
    after=$(<"$counter_path")

    if [[ ! "$after" =~ ^[0-9]+$ ]]; then
      echo "Skipping $label: counter value is not numeric after sampling: $counter_path" >&2
      continue
    fi

    python3 -c 'import sys
dev, rate_name, after, before, window, multiplier = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]), int(sys.argv[6])
print(f"{dev} {rate_name}: {(after - before) * multiplier / window / 1e9:.1f} GB/s")' \
      "$label" "$rate_name" "$after" "$before" "$WINDOW" "$multiplier"

    BEFORE[$i]=$after
  done
done
