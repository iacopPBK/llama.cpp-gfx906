#!/bin/bash

MI50_POWER=225
MI50_SCLK=2000
MI50_MCLK=1125
MI50_CARD=1

DEVICE="/sys/class/drm/card${MI50_CARD}/device"

UPP_VENV="/home/iacopo/upp"
UPP_PYTHON="${UPP_VENV}/bin/python3"

if [ ! -f "$UPP_PYTHON" ]; then
    echo "✗ Error: UPP Python virtual environment not found at $UPP_VENV"
    echo ""
    echo "Please install UPP or update the UPP_VENV variable in this script."
    echo ""
    exit 1
fi

echo "============================================"
echo "AMD MI50 Overclock using UPP"
echo "============================================"
echo "Target Settings:"
echo "  Core Clock (Max): ${MI50_SCLK}MHz"
echo "  Memory Clock (Max): ${MI50_MCLK}MHz"
echo "  Power Limit: ${MI50_POWER}W"
echo "============================================"
echo ""

if [ ! -w "${DEVICE}/pp_table" ]; then
    echo "⚠ Warning: ${DEVICE}/pp_table is not writable"
    echo "You may need to add amdgpu.ppfeaturemask=0xffffffff to kernel parameters"
    echo "in /etc/default/grub (GRUB_CMDLINE_LINUX variable)"
    echo ""
fi

echo "Applying overclock and power settings..."

sudo "$UPP_PYTHON" -m upp.upp -p ${DEVICE}/pp_table set --write \
    smcPPTable/SocketPowerLimitAc0=$MI50_POWER \
    smcPPTable/SocketPowerLimitDc=$MI50_POWER

if [ $? -ne 0 ]; then
    echo "⚠ Warning: Failed to set socket power limits (may not be critical)"
fi

sudo "$UPP_PYTHON" -m upp.upp -p ${DEVICE}/pp_table set --write \
    smcPPTable/FreqTableGfx/8=$MI50_SCLK \
    smcPPTable/FreqTableUclk/2=$MI50_MCLK \
    smcPPTable/FreqTableUclk/3=$MI50_MCLK

if [ $? -ne 0 ]; then
    echo "⚠ Warning: Failed to set clock frequencies"
fi

echo ""
echo "✓ Overclock settings applied successfully!"
echo ""
echo "Applied settings:"
echo "  - Target Core Clock: ${MI50_SCLK}MHz"
echo "  - Target Memory Clock: ${MI50_MCLK}MHz"
echo "  - Power Limit: ${MI50_POWER}W"
echo ""
