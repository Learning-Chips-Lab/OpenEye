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
