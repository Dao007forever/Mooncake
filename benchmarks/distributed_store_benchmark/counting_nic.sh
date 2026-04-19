DEVS=(mlx5_0 mlx5_1)
WINDOW=${1:-5}

declare -A BEFORE
for d in "${DEVS[@]}"; do
  BEFORE[$d]=$(cat /sys/class/infiniband/$d/ports/1/counters/port_rcv_data)
done

sleep "$WINDOW"

for d in "${DEVS[@]}"; do
  after=$(cat /sys/class/infiniband/$d/ports/1/counters/port_rcv_data)
  before=${BEFORE[$d]}
  python3 -c "print(f'$d rcv_rate: {($after-$before)*4/$WINDOW/1e9:.1f} GB/s')"
done
