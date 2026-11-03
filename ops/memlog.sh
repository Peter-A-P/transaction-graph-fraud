# Every five minutes, the host memory available and the card's memory and use, into the run log
# the laptop copies back, so a pod that goes dark leaves its last readings behind.
while true; do
  echo "mem $(date -u +%H:%M) $(free -m | awk '/Mem:/{print $7}')MB available, gpu $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader)" >> /root/run.log
  sleep 300
done
