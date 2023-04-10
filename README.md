# ray-search

```sh
ray start \
      --head \
      --port=6379 \
      --disable-usage-stats \
      --object-manager-port=8076 \
      --include-dashboard=true \
      --dashboard-host=0.0.0.0 \
      --dashboard-port=8266
ray start --head --include-dashboard=true --disable-usage-stats --dashboard-port 9090 --node-ip-address=0.0.0.0
```