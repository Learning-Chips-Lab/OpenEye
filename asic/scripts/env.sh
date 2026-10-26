# Environment for LibreLane runs on IHP SG13G2.
# Source this inside the iic-osic-tools container before invoking librelane:
#   source /foss/designs/OpenEye/asic/scripts/env.sh
#
# PDK/STD_CELL_LIBRARY/PDK_ROOT are already defaults in the container profile;
# set them explicitly here for reproducibility.
export PDK_ROOT=/foss/pdks
export PDK=ihp-sg13g2
export STD_CELL_LIBRARY=sg13g2_stdcell
# Pad library for the automatic padring (sg13g2_IOPad* cells).
export PAD_CELL_LIBRARY=sg13g2_io
# LibreLane 3.1's Odbpy steps invoke openroad with `-python`, which the
# container's default build (/foss/tools/bin/openroad, 26Q4-407) does NOT
# accept (prints usage and exits). The `-python`-capable build ships as
# openroad-librelane (26Q1-1024). Pin it for all openroad steps via
# LibreLane's documented override. (p2_11 accidentally ran on this build
# through a transient $DESIGNS/.designinit PATH override that later
# disappeared -- pinning it here makes the flow self-contained.)
export _LLN_OVERRIDE_OPENROAD=/foss/tools/openroad-librelane/bin/openroad
