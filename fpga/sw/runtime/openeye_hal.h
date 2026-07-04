// This file is part of the OpenEye project.
// © Fachhochschule Dortmund – University of Applied Sciences and Arts (until 2025), Universität Duisburg-Essen (since 2025).
// SPDX-License-Identifier: SHL-2.1
// For more details, see the LICENSE file in the root directory of this project.

// Hardware abstraction for the OpenEye DMA stream interface.
//
// The runtime is platform-agnostic; all platform dependence lives behind
// these four callbacks. Implementations exist for:
//   hal_xaxidma.c     bare-metal Zynq (Xilinx XAxiDma, polling)
//   hal_mock.c        host-side testing (loopback against a reference)
//
// Contract:
//   - dma_write blocks until the accelerator has accepted the full buffer.
//   - The accelerator starts emitting results while a transfer is still
//     being written, so a readback must be armed BEFORE the final write:
//     dma_read_start arms the read (non-blocking), dma_read_wait blocks
//     until it completes. Implementations own cache maintenance.
//   - All return 0 on success, negative on error.

#ifndef OPENEYE_HAL_H
#define OPENEYE_HAL_H

#include <stdint.h>

typedef struct openeye_hal {
    int (*dma_write)(void *ctx, const uint8_t *buf, uint32_t len);
    int (*dma_read_start)(void *ctx, uint8_t *buf, uint32_t len);
    int (*dma_read_wait)(void *ctx);
    void *ctx;
} openeye_hal;

#endif // OPENEYE_HAL_H
