// This file is part of the OpenEye project.
// © Fachhochschule Dortmund – University of Applied Sciences and Arts (until 2025), Universität Duisburg-Essen (since 2025).
// SPDX-License-Identifier: SHL-2.1
// For more details, see the LICENSE file in the root directory of this project.

// Host-side mock HAL for testing the runtime without hardware.
//
// dma_write records how many bytes were "sent"; the armed read is
// satisfied from a caller-supplied response buffer (e.g. the artifact's
// embedded reference), honoring the arm-before-write discipline: reading
// before a read was armed, or arming twice, is an error.

#include "hal_mock.h"

static int mock_write(void *ctx, const uint8_t *buf, uint32_t len) {
    hal_mock_state *s = (hal_mock_state *)ctx;
    (void)buf;
    s->bytes_written += len;
    s->n_writes++;
    if (s->read_armed) {
        /* Transfer that triggers readback: fill the armed buffer now,
         * emulating the accelerator streaming results during the write. */
        if (s->rx_len > s->response_len) return -1;
        for (uint32_t i = 0; i < s->rx_len; i++)
            s->rx_buf[i] = s->response[i];
        s->read_done = 1;
    }
    return 0;
}

static int mock_read_start(void *ctx, uint8_t *buf, uint32_t len) {
    hal_mock_state *s = (hal_mock_state *)ctx;
    if (s->read_armed) return -1; /* double arm */
    s->rx_buf = buf;
    s->rx_len = len;
    s->read_armed = 1;
    s->read_done = 0;
    return 0;
}

static int mock_read_wait(void *ctx) {
    hal_mock_state *s = (hal_mock_state *)ctx;
    if (!s->read_armed || !s->read_done) return -1;
    s->read_armed = 0;
    s->n_reads++;
    return 0;
}

void hal_mock_init(openeye_hal *hal, hal_mock_state *state,
                   const uint8_t *response, uint32_t response_len) {
    state->response = response;
    state->response_len = response_len;
    state->bytes_written = 0;
    state->n_writes = 0;
    state->n_reads = 0;
    state->read_armed = 0;
    state->read_done = 0;
    state->rx_buf = 0;
    state->rx_len = 0;

    hal->dma_write = mock_write;
    hal->dma_read_start = mock_read_start;
    hal->dma_read_wait = mock_read_wait;
    hal->ctx = state;
}
