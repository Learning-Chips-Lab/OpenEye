// This file is part of the OpenEye project.
// © Fachhochschule Dortmund – University of Applied Sciences and Arts (until 2025), Universität Duisburg-Essen (since 2025).
// SPDX-License-Identifier: SHL-2.1
// For more details, see the LICENSE file in the root directory of this project.

#ifndef OPENEYE_HAL_MOCK_H
#define OPENEYE_HAL_MOCK_H

#include "openeye_hal.h"

typedef struct {
    const uint8_t *response;   /* data returned on readback */
    uint32_t response_len;
    uint32_t bytes_written;
    uint32_t n_writes;
    uint32_t n_reads;
    int read_armed;
    int read_done;
    uint8_t *rx_buf;
    uint32_t rx_len;
} hal_mock_state;

void hal_mock_init(openeye_hal *hal, hal_mock_state *state,
                   const uint8_t *response, uint32_t response_len);

#endif /* OPENEYE_HAL_MOCK_H */
