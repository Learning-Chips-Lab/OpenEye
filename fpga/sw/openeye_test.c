#include <stdio.h>
#include <stdlib.h>

#include "platform.h"
#include "xaxidma.h"
#include "xparameters.h"
#include "xbasic_types.h"

#include "dma_stream.h"
#include "dma_stream_out.h"

void memdump(u8 *ptr, u16 length) {
    printf("mem @ %08lx:", (u64) ptr);
    for (int i = 0; i < length; i++) {
        if (i % 16 == 0) {
            puts("");
            printf("%08lx ", (u64) ptr+i);
        }

        if (ptr[i] == 0) {
            printf("\033[38;5;247m%02x \033[0m", ptr[i]);
        } else {
            printf("%02x ", ptr[i]);
        }
    }
    puts("");
    puts("#########################");
}

int configure_dma(u16 axi_dma_id, XAxiDma *axi_dma) {
    XAxiDma_Config *axi_dma_cfg;
    axi_dma_cfg = XAxiDma_LookupConfig(axi_dma_id);
    if (!axi_dma_cfg) {
        printf("No config found for %d\n", axi_dma_id);
        return 1;
    }

    int status = XAxiDma_CfgInitialize(axi_dma, axi_dma_cfg);
    if (status != XST_SUCCESS) {
        printf("Initialization failed for %d with status %d \n", axi_dma_id, status);
        return 1;
    }

    if(XAxiDma_HasSg(axi_dma)){
        printf("Device configured as SG mode\n");
        return 1;
    }

    // disable interrupts -> using polling mode
    XAxiDma_IntrDisable(axi_dma, XAXIDMA_IRQ_ALL_MASK, XAXIDMA_DEVICE_TO_DMA);
    XAxiDma_IntrDisable(axi_dma, XAXIDMA_IRQ_ALL_MASK, XAXIDMA_DMA_TO_DEVICE);

    return 0;
}

int axi_dma_w_transfer(XAxiDma *axi_dma, u8 *tx_data, u32 length) {
    int status;
    Xil_DCacheFlushRange((UINTPTR)tx_data, length);

    // start DMA write transfer
    status = XAxiDma_SimpleTransfer(axi_dma,(UINTPTR) tx_data, length, XAXIDMA_DMA_TO_DEVICE);
    if (status != XST_SUCCESS) {
        printf("Failed TX\r\n");
        return XST_FAILURE;
    }

    // wait for write transfer
    while (XAxiDma_Busy(axi_dma, XAXIDMA_DMA_TO_DEVICE));

    return XST_SUCCESS;
}

int axi_dma_rw_transfer(XAxiDma *axi_dma, u8 *tx_data, u8 *rx_data, u32 length) {
    int status;
    Xil_DCacheFlushRange((UINTPTR)tx_data, length);
    Xil_DCacheFlushRange((UINTPTR)rx_data, length);

    // start DMA read transfer
    status = XAxiDma_SimpleTransfer(axi_dma,(UINTPTR) rx_data, length, XAXIDMA_DEVICE_TO_DMA);
    if (status != XST_SUCCESS) {
        printf("Failed RX %d\r\n", status);
        return XST_FAILURE;
    }

    // start DMA write transfer
    status = XAxiDma_SimpleTransfer(axi_dma,(UINTPTR) tx_data, length, XAXIDMA_DMA_TO_DEVICE);
    if (status != XST_SUCCESS) {
        printf("Failed TX\r\n");
        return XST_FAILURE;
    }

    // wait for write transfer
    while (XAxiDma_Busy(axi_dma, XAXIDMA_DMA_TO_DEVICE));

    // wait for read transfer
    while (XAxiDma_Busy(axi_dma, XAXIDMA_DEVICE_TO_DMA));

    Xil_DCacheFlushRange((UINTPTR)rx_data, length);

    return XST_SUCCESS;
}

int main() {
    int status;
    int error;
    XAxiDma axi_dma_rw;

    init_platform();

    // disable stdout buffering
    setvbuf(stdout, NULL, _IONBF, 0);

    // initialize DMA device
    configure_dma(XPAR_AXIDMA_0_DEVICE_ID, &axi_dma_rw);

    // output buffer
    u8 *data_dma_o = calloc(0x100000, sizeof(u8));

    puts("--- OpenEye Test Start ---");
    for (int i = 0; i < 10; i++) {
        printf("started run %d\n", i);

        // overwrite output buffer, when testing multiple runs
        memset(data_dma_o, 0xcc, dma_output_length);

        // write streams for N-1 layers of the model
        int idx = 0;
        for (int j = 0; j < n_layers - 1; j++) {
            status = axi_dma_w_transfer(&axi_dma_rw, &dma_input_stream[idx], dma_input_stream_lengths[j]);
            if (status != XST_SUCCESS) {
                printf("failed write of layer %d\n", j);
                return status;
            }
            idx += dma_input_stream_lengths[j];
        }

        // write stream for last layer and retrieve results
        status = axi_dma_rw_transfer(&axi_dma_rw, &dma_input_stream[idx], data_dma_o, dma_input_stream_lengths[n_layers-1]);
        if (status != XST_SUCCESS) {
            puts("failed read/write of last layer");
            return status;
        }

        // compare results with reference stream
        int n_values = dma_output_length / 8;
        printf("checking %d values\n", n_values);
        error = 0;
        for (u32 i = 0; i < n_values; i ++) {
            if (((u64*) data_dma_o)[i] != ((u64*) dma_output_reference)[i]) {
                error += 1;
                if (error && error < 10) {
                    printf("error at index %d: %016lx / %016lx\n", i, ((u64*) data_dma_o)[i], ((u64*) dma_output_reference)[i]);
                }
            }
        }
        if (error == 0) {
            puts("\033[38;5;10mOK\033[0m");
        } else {
            printf("\033[38;5;9mBAD - %d errors\033[0m\n", error);
        }
    }

    puts("--- OpenEye Test End ---");
    cleanup_platform();
    return 0;
}
