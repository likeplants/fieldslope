pids=$(sudo lsof -t -i :8003 -sTCP:LISTEN 2>/dev/null)

if [ -n "$pids" ]; then
    echo "Killing process(es) on port 8003: $pids"
    sudo kill -9 $pids
else
    echo "Warning: no process found on port 8003"
fi
