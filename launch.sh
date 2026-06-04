#!/usr/bin/env bash
set -e

echo ""
echo " ╔══════════════════════════════════════════════╗"
echo " ║   IS Systematic Review  •  AI Assistant      ║"
echo " ║   Vietnam National University — Hanoi        ║"
echo " ╚══════════════════════════════════════════════╝"
echo ""

# ── Check Docker ─────────────────────────────────────────────────────────
if ! command -v docker &>/dev/null; then
    echo " [!] Docker is not installed."
    echo "     Mac:   https://www.docker.com/products/docker-desktop/"
    echo "     Linux: sudo apt install docker.io docker-compose-plugin"
    exit 1
fi

if ! docker info &>/dev/null; then
    echo " [!] Docker is not running. Please start Docker Desktop."
    exit 1
fi

echo " [OK] Docker is running."
echo ""
echo " [>>] Building and starting... (first run ~10-20 min for AI models)"
echo ""

docker compose up --build -d

echo ""
echo " [..] Waiting for application to be ready..."
until curl -sf http://localhost > /dev/null 2>&1; do
    sleep 3
done

echo ""
echo " ╔══════════════════════════════════════════════╗"
echo " ║  Application is ready!                       ║"
echo " ║                                              ║"
echo " ║  Open: http://localhost                      ║"
echo " ║  Stop: docker compose down                   ║"
echo " ╚══════════════════════════════════════════════╝"
echo ""

# Open browser
if command -v xdg-open &>/dev/null; then
    xdg-open http://localhost
elif command -v open &>/dev/null; then
    open http://localhost
fi
