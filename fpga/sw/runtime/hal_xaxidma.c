// This file is part of the OpenEye project.
// © Fachhochschule Dortmund – University of Applied Sciences and Arts (until 2025), Universität Duisburg-Essen (since 2025).
// SPDX-License-Identifier: SHL-2.1
// For more details, see the LICENSE file in the root directory of this project.

// Bare-metal Zynq HAL backend: Xilinx AXI DMA in simple (non-SG) polling
// mode. Same transfer discipline as the original fpga/sw/openeye_test.c.
//
// Build only in a Vitis/standalone BSP environment (needs xaxidma.h).

#include "openeye_hal.h"

#include "xaxidma.h"
#include "xil_cache.h"

typedef struct {
    XAxiDma dma;
    uint8_t *rx_buf;
    uint32_t rx_len;
} hal_xaxidma_state;

static hal_xaxidma_state g_state;

static int xaxidma_write(void *ctx, const uint8_t *buf, uint32_t len) {
    hal_xaxidma_state *s = (hal_xaxidma_state *)ctx;
    Xil_DCacheFlushRange((UINTPTR)buf, len);
    if (XAxiDma_SimpleTransfer(&s->dma, (UINTPTR)buf, len,
                               XAXIDMA_DMA_TO_DEVICE) != XST_SUCCESS)
        return -1;
    while (XAxiDma_Busy(&s->dma, XAXIDMA_DMA_TO_DEVICE))
        ;
    return 0;
}

static int xaxidma_read_start(void *ctx, uint8_t *buf, uint32_t len) {
    hal_xaxidma_state *s = (hal_xaxidma_state *)ctx;
    Xil_DCacheFlushRange((UINTPTR)buf, len);
    if (XAxiDma_SimpleTransfer(&s->dma, (UINTPTR)buf, len,
                               XAXIDMA_DEVICE_TO_DMA) != XST_SUCCESS)
        return -1;
    s->rx_buf = buf;
    s->rx_len = len;
    return 0;
}

static int xaxidma_read_wait(void *ctx) {
    hal_xaxidma_state *s = (hal_xaxidma_state *)ctx;
    if (!s->rx_buf) return -1;
    while (XAxiDma_Busy(&s->dma, XAXIDMA_DEVICE_TO_DMA))
        ;
    /* Results were written by the PL; drop stale cache lines. */
    Xil_DCacheInvalidateRange((UINTPTR)s->rx_buf, s->rx_len);
    s->rx_buf = 0;
    return 0;
}

/* Initialize the AXI DMA device and fill in the HAL. */
int hal_xaxidma_init(openeye_hal *hal, u16 device_id) {
    XAxiDma_Config *cfg = XAxiDma_LookupConfig(device_id);
    if (!cfg) return -1;
    if (XAxiDma_CfgInitialize(&g_state.dma, cfg) != XST_SUCCESS) return -1;
    if (XAxiDma_HasSg(&g_state.dma)) return -1; /* simple mode expected */

    XAxiDma_IntrDisable(&g_state.dma, XAXIDMA_IRQ_ALL_MASK,
                        XAXIDMA_DEVICE_TO_DMA);
    XAxiDma_IntrDisable(&g_state.dma, XAXIDMA_IRQ_ALL_MASK,
                        XAXIDMA_DMA_TO_DEVICE);

    hal->dma_write = xaxidma_write;
    hal->dma_read_start = xaxidma_read_start;
    hal->dma_read_wait = xaxidma_read_wait;
    hal->ctx = &g_state;
    return 0;
}
