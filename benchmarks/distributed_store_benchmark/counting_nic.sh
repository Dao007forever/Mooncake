#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 [window_seconds] [device ...]" >&2
  echo "       DEVS=mlx5_0,mlx5_1 $0 [window_seconds]" >&2
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

PORT=${PORT:-1}
COUNTER=${COUNTER:-port_rcv_data}
SYSFS_INFINIBAND=${SYSFS_INFINIBAND:-/sys/class/infiniband}
USER_DEVS=${DEVS:-}

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
  echo "No InfiniBand devices specified or found under $SYSFS_INFINIBAND" >&2
  usage
  exit 1
fi

ACTIVE_DEVS=()
COUNTER_PATHS=()
BEFORE=()
for d in "${DEVS[@]}"; do
  counter_path="$SYSFS_INFINIBAND/$d/ports/$PORT/counters/$COUNTER"
  if [[ ! -r "$counter_path" ]]; then
    echo "Skipping $d: counter not found or not readable: $counter_path" >&2
    continue
  fi

  before=$(<"$counter_path")
  if [[ ! "$before" =~ ^[0-9]+$ ]]; then
    echo "Skipping $d: counter value is not numeric: $counter_path" >&2
    continue
  fi

  ACTIVE_DEVS+=("$d")
  COUNTER_PATHS+=("$counter_path")
  BEFORE+=("$before")
done

if ((${#ACTIVE_DEVS[@]} == 0)); then
  echo "No readable $COUNTER counters found." >&2
  if [[ -d "$SYSFS_INFINIBAND" ]]; then
    echo "Available devices under $SYSFS_INFINIBAND:" >&2
    for dev_path in "$SYSFS_INFINIBAND"/*; do
      [[ -e "$dev_path" ]] || continue
      echo "  ${dev_path##*/}" >&2
    done
  fi
  usage
  exit 1
fi

sleep "$WINDOW"

for i in "${!ACTIVE_DEVS[@]}"; do
  d=${ACTIVE_DEVS[$i]}
  counter_path=${COUNTER_PATHS[$i]}
  before=${BEFORE[$i]}
  after=$(<"$counter_path")

  if [[ ! "$after" =~ ^[0-9]+$ ]]; then
    echo "Skipping $d: counter value is not numeric after sampling: $counter_path" >&2
    continue
  fi

  python3 -c 'import sys
dev, after, before, window = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
print(f"{dev} rcv_rate: {(after - before) * 4 / window / 1e9:.1f} GB/s")' \
    "$d" "$after" "$before" "$WINDOW"
done
