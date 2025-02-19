#include <stdio.h>
#include <stdlib.h>
#include <sleep.h>

#include "platform.h"
#include "xaxidma.h"
#include "xparameters.h"
#include "xbasic_types.h"

#include "dma_input.h"
#include "dma_output_ref.h"

void memdump(u8 *ptr, u16 length) {
	printf("mem @ %08lx:", (u64) ptr);
    for (int i = 0; i < length; i++) {
        if (i % 16 == 0) {
        	printf("\n%08lx ", (u64) ptr+i);
        }

        if (ptr[i] == 0) {
        	printf("\033[38;5;247m%02x \033[0m", ptr[i]);
        } else {
        	printf("%02x ", ptr[i]);
        }
    }
    puts("\n#########################");
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

int axi_dma_rw_transfer(XAxiDma *axi_dma_w, u8 *tx_data, XAxiDma *axi_dma_r, u8 *rx_data, u32 length) {
	int status;
	Xil_DCacheFlushRange((UINTPTR)tx_data, length);
	Xil_DCacheFlushRange((UINTPTR)rx_data, length);

	// start read Data
	status = XAxiDma_SimpleTransfer(axi_dma_r,(UINTPTR) rx_data, length, XAXIDMA_DEVICE_TO_DMA);
	if (status != XST_SUCCESS) {
		printf("Failed RX %d\r\n", status);
		return XST_FAILURE;
	}

	// start write Data
	status = XAxiDma_SimpleTransfer(axi_dma_w,(UINTPTR) tx_data, length, XAXIDMA_DMA_TO_DEVICE);
	if (status != XST_SUCCESS) {
		printf("Failed TX\r\n");
		return XST_FAILURE;
	}

	// wait for write transfer
	while (XAxiDma_Busy(axi_dma_w, XAXIDMA_DMA_TO_DEVICE));

	// wait for read transfer
	while (XAxiDma_Busy(axi_dma_r, XAXIDMA_DEVICE_TO_DMA));

	Xil_DCacheFlushRange((UINTPTR)rx_data, length);

	return 0;
}

int main() {
	int status;
	int error;
	int data_len;
	int values;

	XAxiDma axi_dma_rw;

	init_platform();

	// disable stdio buffering
	setvbuf(stdout, NULL, _IONBF, 0);

	// initialize DMA device
    configure_dma(XPAR_AXIDMA_0_DEVICE_ID, &axi_dma_rw);

    u8 *data_dma_o = calloc(65536, sizeof(u8));
    data_len = 28176;
    values = 2304;

	puts("");
	for (int i = 0; i < 10; i++) {
        printf("started run %d\n", i);
        memset(data_dma_o, 0xcc, 65536);
        
        // write config and data / read results
        status = axi_dma_rw_transfer(&axi_dma_rw, data_in, &axi_dma_rw, data_dma_o, data_len);
        if (status != XST_SUCCESS) {
            puts("rw failed");
            return status;
        }

        error = 0;
        for (u32 i = 0; i < values; i ++) {
            if (((u64*) data_dma_o)[i] != ((u64*) data_out)[i]) {
                error += 1;
                if (error == 1) {
                    printf("error at index %d: %016lx / %016lx\n", i, ((u64*) data_dma_o)[i], ((u64*) data_out)[i]);
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
