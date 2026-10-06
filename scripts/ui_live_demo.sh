#!/bin/bash
# The panel is the nx1 terminal. SSH there and run:
#   bash /home/jetson/cc/edgevisor_fresh/scripts/ui_live_demo_root.sh
exec ssh -t -o BatchMode=yes -o ConnectTimeout=20 jetson@10.47.145.51 'bash /home/jetson/cc/edgevisor_fresh/scripts/ui_live_demo_root.sh'
