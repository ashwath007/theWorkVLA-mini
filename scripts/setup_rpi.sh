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

# ── 6. systemd service ────────────────────────────────────────────────────────
info "Installing systemd service …"
SERVICE_FILE="/etc/systemd/system/vla-recorder.service"

cat > "$SERVICE_FILE" << EOF
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

systemctl daemon-reload
systemctl enable vla-recorder.service
success "systemd service installed and enabled."

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
echo "  Service        : vla-recorder.service"
echo ""
echo "  To start recording:"
echo "    sudo systemctl start vla-recorder"
echo ""
echo "  Or manually:"
echo "    source $VENV_DIR/bin/activate"
echo "    vla-record record --duration 3600"
echo ""
warn "A REBOOT is required to activate I2C interface."
echo ""
