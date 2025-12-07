#!/bin/bash
cat << 'EOF'

   ██╗     ██╗      █████╗ ███╗   ███╗ █████╗    ██████╗██████╗ ██████╗
   ██║     ██║     ██╔══██╗████╗ ████║██╔══██╗  ██╔════╝██╔══██╗██╔══██╗
   ██║     ██║     ███████║██╔████╔██║███████║  ██║     ██████╔╝██████╔╝
   ██║     ██║     ██╔══██║██║╚██╔╝██║██╔══██║  ██║     ██╔═══╝ ██╔═══╝
   ███████╗███████╗██║  ██║██║ ╚═╝ ██║██║  ██║  ╚██████╗██║     ██║
   ╚══════╝╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝   ╚═════╝╚═╝     ╚═╝

    ██████╗ ███████╗██╗  ██╗ █████╗  ██████╗  ██████╗
   ██╔════╝ ██╔════╝╚██╗██╔╝██╔══██╗██╔═████╗██╔════╝
   ██║  ███╗█████╗   ╚███╔╝ ╚██████║██║██╔██║███████╗
   ██║   ██║██╔══╝   ██╔██╗  ╚═══██║████╔╝██║██╔═══██╗
   ╚██████╔╝██║     ██╔╝ ██╗ █████╔╝╚██████╔╝╚██████╔╝
    ╚═════╝ ╚═╝     ╚═╝  ╚═╝ ╚════╝  ╚═════╝  ╚═════╝

EOF

MI50_POWER=225
MI50_SCLK=2000
MI50_MCLK=1100
MI50_CARD=1

DEVICE="/sys/class/drm/card${MI50_CARD}/device"
UPP_PYTHON="/home/iacoppbk/upp/bin/python3"

[ ! -f "$UPP_PYTHON" ] && echo "Error: UPP not found at $UPP_PYTHON" && exit 1

echo "Applying: Core=${MI50_SCLK}MHz, Mem=${MI50_MCLK}MHz, Power=${MI50_POWER}W"

sudo "$UPP_PYTHON" -m upp.upp -p ${DEVICE}/pp_table set --write \
    smcPPTable/SocketPowerLimitAc0=$MI50_POWER \
    smcPPTable/SocketPowerLimitDc=$MI50_POWER

sudo "$UPP_PYTHON" -m upp.upp -p ${DEVICE}/pp_table set --write \
    smcPPTable/FreqTableGfx/8=$MI50_SCLK \
    smcPPTable/FreqTableUclk/2=$MI50_MCLK \
    smcPPTable/FreqTableUclk/3=$MI50_MCLK

echo "Done"
