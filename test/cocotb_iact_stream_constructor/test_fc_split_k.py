# This file is part of the OpenEye project.
# © Fachhochschule Dortmund – University of Applied Sciences and Arts (until 2025), Universität Duisburg-Essen (since 2025).
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.

"""Dense converter regression: bank distribution, padding and K-tile readout."""
import os

import cocotb_test.simulator
import pytest
from open_eye import hdl_dir, test_dir


@pytest.mark.parametrize('cluster_rows', [1, 2])
@pytest.mark.parametrize('channels', [2, 6, 10, 12])
@pytest.mark.parametrize('gaps', [0, 1], ids=['continuous', 'gapped'])
@pytest.mark.parametrize('banks,pe_rows',
                         [(1, 3), (2, 3), (3, 3), (4, 3), (1, 4), (4, 1)])
def test_fc_split_k(cluster_rows, channels, gaps, banks, pe_rows):
    for row in range(cluster_rows):
        cocotb_test.simulator.run(
            simulator='icarus',
            verilog_sources=[os.path.join(hdl_dir, name) for name in
                             ('iact_stream_constructor.v', 'RAM_SP.v', 'RAM_SP_generic.v')],
            toplevel='iact_stream_constructor', module='fc_split_k_tb',
            python_search=[os.path.dirname(__file__)],
            sim_build=os.path.join(test_dir, '.temp',
                                   f'fc_split_k_{cluster_rows}_{channels}_{row}_{gaps}_{banks}_{pe_rows}'),
            parameters={'CLUSTER_ROWS': cluster_rows, 'CLUSTER_ROW_ID': row,
                        'NUM_GLB_IACT': banks, 'PE_Y': pe_rows,
                        'WORD_BITWIDTH': 24 * banks, 'ADDRWIDTH': 6},
            extra_env={'FC_ROWS': str(cluster_rows), 'FC_ROW': str(row),
                       'FC_CHANNELS': str(channels), 'FC_GAPS': str(gaps),
                       'FC_BANKS': str(banks), 'FC_PE_ROWS': str(pe_rows)},
        )
