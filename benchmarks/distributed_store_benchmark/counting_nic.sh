#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 [window_seconds] [device ...]" >&2
  echo "       DEVS=mlx5_0,mlx5_1 $0 [window_seconds]" >&2
  echo "       MODE=auto|roce|ib PORT=1 GID_INDEX=3 $0 [window_seconds] [device ...]" >&2
  echo "       PORT defaults to all ports; set PORT=1 to restrict it." >&2
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

PORT=${PORT:-}
MODE=${MODE:-auto}
GID_INDEX=${GID_INDEX:-}
IB_COUNTER=${IB_COUNTER:-${COUNTER:-}}
NET_COUNTER=${NET_COUNTER:-rx_bytes}
SYSFS_INFINIBAND=${SYSFS_INFINIBAND:-/sys/class/infiniband}
SYSFS_NET=${SYSFS_NET:-/sys/class/net}
USER_DEVS=${DEVS:-}
IB_COUNTERS=()

case "$MODE" in
  auto | roce | ib) ;;
  *)
    echo "MODE must be one of: auto, roce, ib" >&2
    usage
    exit 2
    ;;
esac

if [[ -n "$GID_INDEX" && ! "$GID_INDEX" =~ ^[0-9]+$ ]]; then
  echo "GID_INDEX must be a non-negative integer: $GID_INDEX" >&2
  usage
  exit 2
fi

if [[ -n "$PORT" && ! "$PORT" =~ ^[0-9]+$ ]]; then
  echo "PORT must be a non-negative integer: $PORT" >&2
  usage
  exit 2
fi

if [[ -n "$IB_COUNTER" ]]; then
  IB_COUNTERS=("$IB_COUNTER")
elif [[ -n "${COUNTERS:-}" ]]; then
  read -r -a IB_COUNTERS <<< "${COUNTERS//,/ }"
else
  IB_COUNTERS=(port_rcv_data port_xmit_data)
fi

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
else
  DEVS=(mlx5_0 mlx5_1)
fi

if ((${#DEVS[@]} == 0)); then
  echo "No RDMA devices specified or found under $SYSFS_INFINIBAND" >&2
  usage
  exit 1
fi

ACTIVE_LABELS=()
ACTIVE_RATE_NAMES=()
COUNTER_PATHS=()
BEFORE=()
MULTIPLIERS=()
ROCE_NETDEVS=()

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

add_roce_netdev() {
  local ndev=$1
  local existing

  [[ -n "$ndev" ]] || return 0
  for existing in "${ROCE_NETDEVS[@]:-}"; do
    [[ -n "$existing" ]] || continue
    [[ "$existing" == "$ndev" ]] && return 0
  done

  ROCE_NETDEVS+=("$ndev")
}

add_roce_netdev_from_file() {
  local ndev_file=$1
  local ndev

  ndev=$(cat "$ndev_file" 2>/dev/null || true)
  ndev=${ndev%%$'\n'*}
  add_roce_netdev "$ndev"
}

collect_roce_netdevs_from_port() {
  local d=$1
  local port=$2
  local ndev_dir
  local ndev_file

  ndev_dir="$SYSFS_INFINIBAND/$d/ports/$port/gid_attrs/ndevs"
  if [[ -n "$GID_INDEX" ]]; then
    ndev_file="$ndev_dir/$GID_INDEX"
    [[ -r "$ndev_file" ]] && add_roce_netdev_from_file "$ndev_file"
    return 0
  fi

  if [[ -d "$ndev_dir" ]]; then
    for ndev_file in "$ndev_dir"/*; do
      [[ -r "$ndev_file" ]] || continue
      add_roce_netdev_from_file "$ndev_file"
    done
  fi

  return 0
}

collect_roce_netdevs_from_rdma() {
  local d=$1
  local port_dir

  if [[ -n "$PORT" ]]; then
    collect_roce_netdevs_from_port "$d" "$PORT"
    return 0
  fi

  if [[ -d "$SYSFS_INFINIBAND/$d/ports" ]]; then
    for port_dir in "$SYSFS_INFINIBAND/$d/ports"/*; do
      [[ -d "$port_dir" ]] || continue
      collect_roce_netdevs_from_port "$d" "${port_dir##*/}"
    done
  fi
}

collect_roce_netdevs_from_net() {
  local d=$1
  local ib_path
  local net_path

  for net_path in "$SYSFS_NET"/*; do
    [[ -d "$net_path" ]] || continue
    [[ -r "$net_path/statistics/$NET_COUNTER" ]] || continue
    for ib_path in "$net_path"/device/infiniband/*; do
      [[ -e "$ib_path" ]] || continue
      [[ "${ib_path##*/}" == "$d" ]] && add_roce_netdev "${net_path##*/}"
    done
  done
}

collect_roce_netdevs() {
  local d=$1
  local dev_net_dir
  local net_path

  ROCE_NETDEVS=()

  if [[ -r "$SYSFS_NET/$d/statistics/$NET_COUNTER" ]]; then
    add_roce_netdev "$d"
    return 0
  fi

  collect_roce_netdevs_from_rdma "$d"

  dev_net_dir="$SYSFS_INFINIBAND/$d/device/net"
  if [[ -d "$dev_net_dir" ]]; then
    for net_path in "$dev_net_dir"/*; do
      [[ -e "$net_path" ]] || continue
      add_roce_netdev "${net_path##*/}"
    done
  fi

  collect_roce_netdevs_from_net "$d"

  return 0
}

add_roce_samples() {
  local d=$1
  local quiet=${2:-0}
  local before_count
  local label
  local ndev

  before_count=${#ACTIVE_LABELS[@]}
  collect_roce_netdevs "$d"

  if ((${#ROCE_NETDEVS[@]} == 0)); then
    if ((quiet == 0)); then
      if [[ -n "$PORT" ]]; then
        echo "Skipping $d: no RoCE netdev mapping found for port $PORT" >&2
      else
        echo "Skipping $d: no RoCE netdev mapping found" >&2
      fi
    fi
    return 1
  fi

  for ndev in "${ROCE_NETDEVS[@]}"; do
    label=$ndev
    if [[ "$ndev" != "$d" ]]; then
      label="$d/$ndev"
    fi
    add_sample "$label" "$SYSFS_NET/$ndev/statistics/$NET_COUNTER" 1 "RoCE" "$(rate_name_for_counter "$NET_COUNTER")" || true
  done

  ((${#ACTIVE_LABELS[@]} > before_count))
}

add_ib_sample() {
  local d=$1
  local added=1
  local port_dir
  local port
  local counter
  local label

  if [[ -n "$PORT" ]]; then
    for counter in "${IB_COUNTERS[@]}"; do
      label="$d"
      if ((${#IB_COUNTERS[@]} > 1)); then
        label="$d/$counter"
      fi
      add_sample "$label" "$SYSFS_INFINIBAND/$d/ports/$PORT/counters/$counter" 4 "InfiniBand" "$(rate_name_for_counter "$counter")" && added=0
    done
    return "$added"
  fi

  if [[ -d "$SYSFS_INFINIBAND/$d/ports" ]]; then
    for port_dir in "$SYSFS_INFINIBAND/$d/ports"/*; do
      [[ -d "$port_dir" ]] || continue
      port=${port_dir##*/}
      for counter in "${IB_COUNTERS[@]}"; do
        label="$d/port$port"
        if ((${#IB_COUNTERS[@]} > 1)); then
          label="$label/$counter"
        fi
        add_sample "$label" "$port_dir/counters/$counter" 4 "InfiniBand" "$(rate_name_for_counter "$counter")" && added=0
      done
    done
  fi

  return "$added"
}

for d in "${DEVS[@]}"; do
  case "$MODE" in
    roce)
      add_roce_samples "$d" || true
      ;;
    ib)
      add_ib_sample "$d" || true
      ;;
    auto)
      if ! add_roce_samples "$d" 1; then
        add_ib_sample "$d" || true
      fi
      ;;
  esac
done

if ((${#ACTIVE_LABELS[@]} == 0)); then
  echo "No readable counters found." >&2
  echo "Searched RoCE counters under $SYSFS_NET/<netdev>/statistics/$NET_COUNTER" >&2
  if [[ -n "$PORT" ]]; then
    echo "Searched InfiniBand counters under $SYSFS_INFINIBAND/<dev>/ports/$PORT/counters/{${IB_COUNTERS[*]}}" >&2
  else
    echo "Searched InfiniBand counters under $SYSFS_INFINIBAND/<dev>/ports/<port>/counters/{${IB_COUNTERS[*]}}" >&2
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
      [[ -r "$net_path/statistics/$NET_COUNTER" ]] || continue
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
