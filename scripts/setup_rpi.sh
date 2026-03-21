#!/usr/bin/env bash
# ============================================================================
# setup_rpi.sh — Set up Raspberry Pi for India VLA headset recording
# ============================================================================
# Usage:  sudo bash setup_rpi.sh
# Tested: Raspberry Pi OS (Bullseye / Bookworm), Python 3.11
# ============================================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

info()    { echo -e "${CYAN}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

# ── Require root ──────────────────────────────────────────────────────────────
[[ $EUID -ne 0 ]] && error "Please run as root: sudo bash setup_rpi.sh"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="${DATA_DIR:-/data/sessions}"
SERVICE_USER="${SERVICE_USER:-pi}"

info "Project root: $PROJECT_ROOT"
info "Data directory: $DATA_DIR"

# ── 1. System packages ────────────────────────────────────────────────────────
info "Installing system packages …"
apt-get update -qq
apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    ffmpeg \
    libopencv-dev \
    python3-opencv \
    libsndfile1 \
    portaudio19-dev \
    libatlas-base-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    i2c-tools \
    python3-smbus \
    git \
    curl \
    htop \
    screen

success "System packages installed."

# ── 2a. Enable UART for NEO-M8N GPS ──────────────────────────────────────────
info "Enabling UART for GPS receiver …"

_add_config_line() {
    local file="$1"
    local line="$2"
    if [[ -f "$file" ]] && ! grep -q "^${line}" "$file"; then
        echo "$line" >> "$file"
        success "Added '$line' to $file"
    fi
}

for CONFIG_FILE in /boot/config.txt /boot/firmware/config.txt; do
    [[ -f "$CONFIG_FILE" ]] || continue
    _add_config_line "$CONFIG_FILE" "enable_uart=1"
    _add_config_line "$CONFIG_FILE" "dtoverlay=disable-bt"
done

# Disable the serial console so the GPS module can use /dev/ttyAMA0
if grep -q "console=serial0" /boot/cmdline.txt 2>/dev/null; then
    sed -i 's/console=serial0,[0-9]* //g' /boot/cmdline.txt
    success "Removed serial console from /boot/cmdline.txt"
fi
if [[ -f /boot/firmware/cmdline.txt ]]; then
    if grep -q "console=serial0" /boot/firmware/cmdline.txt; then
        sed -i 's/console=serial0,[0-9]* //g' /boot/firmware/cmdline.txt
        success "Removed serial console from /boot/firmware/cmdline.txt"
    fi
fi

# Disable Bluetooth modem service (it occupies UART0 on Pi 3/4/5)
systemctl disable hciuart 2>/dev/null || true
systemctl stop    hciuart 2>/dev/null || true
success "UART enabled for GPS (enable_uart=1, dtoverlay=disable-bt)."

# ── 2. Enable I2C for MPU-9250 ────────────────────────────────────────────────
info "Enabling I2C interface …"
if grep -q "^#dtparam=i2c_arm=on" /boot/config.txt 2>/dev/null; then
    sed -i 's/^#dtparam=i2c_arm=on/dtparam=i2c_arm=on/' /boot/config.txt
    success "I2C enabled in /boot/config.txt (was commented out)."
elif ! grep -q "^dtparam=i2c_arm=on" /boot/config.txt 2>/dev/null; then
    echo "dtparam=i2c_arm=on" >> /boot/config.txt
    success "I2C enabled in /boot/config.txt (appended)."
else
    info "I2C already enabled."
fi

# Also handle Bookworm (config.txt → /boot/firmware/config.txt)
if [[ -f /boot/firmware/config.txt ]]; then
    if ! grep -q "^dtparam=i2c_arm=on" /boot/firmware/config.txt; then
        echo "dtparam=i2c_arm=on" >> /boot/firmware/config.txt
        success "I2C enabled in /boot/firmware/config.txt."
    fi
fi

modprobe i2c-dev 2>/dev/null || true
if [[ -e /dev/i2c-1 ]]; then
    usermod -aG i2c "$SERVICE_USER" 2>/dev/null || true
    success "User $SERVICE_USER added to i2c group."
fi

# ── 3. Python virtual environment ─────────────────────────────────────────────
VENV_DIR="$PROJECT_ROOT/.venv"
info "Creating Python virtual environment at $VENV_DIR …"
python3.11 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

pip install --upgrade pip wheel setuptools

info "Installing Python dependencies …"
pip install -r "$PROJECT_ROOT/requirements.txt"

# RPi extras
pip install smbus2 RPi.GPIO 2>/dev/null || warn "RPi.GPIO not installed (may need reboot)."

# GPS serial support
info "Installing GPS serial libraries …"
pip install pyserial pynmea2 || warn "pyserial / pynmea2 installation failed."

# HTTP upload client for ChunkStreamer
pip install requests 2>/dev/null || warn "requests not installed."

deactivate
success "Python environment ready."

# ── 4. Data directory structure ───────────────────────────────────────────────
info "Creating data directories …"
mkdir -p "$DATA_DIR"
mkdir -p /models
mkdir -p /var/log/vla

chown -R "$SERVICE_USER:$SERVICE_USER" "$DATA_DIR" /models /var/log/vla 2>/dev/null || true
success "Data directories created."

# ── 5. .env file ─────────────────────────────────────────────────────────────
ENV_FILE="$PROJECT_ROOT/.env"
if [[ ! -f "$ENV_FILE" ]]; then
    cp "$PROJECT_ROOT/.env.example" "$ENV_FILE"
    sed -i "s|/data/sessions|$DATA_DIR|g" "$ENV_FILE"
    success ".env file created from .env.example."
fi

# ── 6. systemd services ───────────────────────────────────────────────────────
info "Installing systemd services …"

# ── 6a. Recorder service ──────────────────────────────────────────────────────
RECORDER_SERVICE_FILE="/etc/systemd/system/vla-recorder.service"

cat > "$RECORDER_SERVICE_FILE" << EOF
[Unit]
Description=India VLA Headset Recorder
After=network.target
Wants=network.target

[Service]
Type=simple
User=$SERVICE_USER
WorkingDirectory=$PROJECT_ROOT
EnvironmentFile=$PROJECT_ROOT/.env
ExecStart=$VENV_DIR/bin/python -m src.capture.cli record --simulate-imu
Restart=on-failure
RestartSec=5
StandardOutput=append:/var/log/vla/recorder.log
StandardError=append:/var/log/vla/recorder.log

[Install]
WantedBy=multi-user.target
EOF

success "Recorder systemd service written: $RECORDER_SERVICE_FILE"

# ── 6b. Streamer service ──────────────────────────────────────────────────────
# The streamer watches the session directory and uploads chunks to the server.
# Set VLA_SERVER_URL and VLA_API_KEY in /etc/default/vla-streamer or the .env file.

STREAMER_SERVICE_FILE="/etc/systemd/system/vla-streamer.service"
STREAMER_ENV_FILE="/etc/default/vla-streamer"

# Write a defaults file if it doesn't exist
if [[ ! -f "$STREAMER_ENV_FILE" ]]; then
    cat > "$STREAMER_ENV_FILE" << 'ENVEOF'
# VLA Chunk Streamer environment
# Set these before starting the service.
VLA_SERVER_URL=http://192.168.1.100:8000
VLA_SESSION_ID=default-session
VLA_DATA_DIR=/data/sessions
VLA_API_KEY=
VLA_DB_PATH=/tmp/vla_upload_queue.db
ENVEOF
    success "Streamer defaults written to $STREAMER_ENV_FILE"
fi

cat > "$STREAMER_SERVICE_FILE" << EOF
[Unit]
Description=India VLA Chunk Streamer
Documentation=https://github.com/india-vla/india-vla-engine
After=network-online.target
Wants=network-online.target
# Start after the recorder so the session directory exists
After=vla-recorder.service

[Service]
Type=simple
User=$SERVICE_USER
WorkingDirectory=$PROJECT_ROOT
EnvironmentFile=$PROJECT_ROOT/.env
EnvironmentFile=-$STREAMER_ENV_FILE
ExecStart=$VENV_DIR/bin/python -c "
import os, time, logging
logging.basicConfig(level=logging.INFO)
from src.capture.streamer import ChunkStreamer

streamer = ChunkStreamer(
    server_url=os.environ.get('VLA_SERVER_URL', 'http://localhost:8000'),
    session_id=os.environ.get('VLA_SESSION_ID', 'default-session'),
    data_dir=os.environ.get('VLA_DATA_DIR', '/data/sessions'),
    api_key=os.environ.get('VLA_API_KEY') or None,
    db_path=os.environ.get('VLA_DB_PATH', '/tmp/vla_upload_queue.db'),
)
streamer.start()

try:
    while True:
        time.sleep(30)
        stats = streamer.get_stats()
        logging.info('Streamer stats: %s', stats)
except KeyboardInterrupt:
    pass
finally:
    streamer.stop()
"
Restart=on-failure
RestartSec=10
StandardOutput=append:/var/log/vla/streamer.log
StandardError=append:/var/log/vla/streamer.log

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable vla-recorder.service
systemctl enable vla-streamer.service
success "Recorder and Streamer systemd services installed and enabled."

# ── 7. Log rotation ───────────────────────────────────────────────────────────
cat > /etc/logrotate.d/vla-recorder << 'EOF'
/var/log/vla/*.log {
    daily
    rotate 7
    compress
    missingok
    notifempty
    sharedscripts
    postrotate
        systemctl reload vla-recorder.service 2>/dev/null || true
    endscript
}
EOF
success "Log rotation configured."

# ── 8. Summary ────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}  India VLA Raspberry Pi setup complete!${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""
echo "  Data directory : $DATA_DIR"
echo "  Virtual env    : $VENV_DIR"
echo "  Services       : vla-recorder.service  vla-streamer.service"
echo ""
echo "  To start recording:"
echo "    sudo systemctl start vla-recorder"
echo ""
echo "  To start the chunk uploader:"
echo "    sudo systemctl start vla-streamer"
echo ""
echo "  Configure the server URL before starting the streamer:"
echo "    sudo nano $STREAMER_ENV_FILE"
echo ""
echo "  Or manually:"
echo "    source $VENV_DIR/bin/activate"
echo "    vla-record record --duration 3600"
echo ""
warn "A REBOOT is required to activate I2C and UART interfaces."
echo ""
