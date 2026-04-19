for d in mlx5_3 mlx5_4; do
  echo -n "$d rcv_rate: "
  a=$(cat /sys/class/infiniband/$d/ports/1/counters/port_rcv_data)
  sleep 2
  b=$(cat /sys/class/infiniband/$d/ports/1/counters/port_rcv_data)
  python3 -c "print(f'{($b-$a)*4/2/1e9:.1f} GB/s')"
done
