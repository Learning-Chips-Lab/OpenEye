// This file is part of the OpenEye project.
// © Fachhochschule Dortmund – University of Applied Sciences and Arts (until 2025), Universität Duisburg-Essen (since 2025).
// SPDX-License-Identifier: SHL-2.1
// For more details, see the LICENSE file in the root directory of this project.

// Bare-metal example: run an .oeye artifact on a Zynq via XAxiDma.
//
// The artifact is expected in DDR at OPENEYE_ARTIFACT_ADDR (loaded via
// JTAG/XSCT, U-Boot, or SD card copy) with its byte length at
// OPENEYE_ARTIFACT_LEN_ADDR, or linked in as a symbol; adapt load_artifact
// to your boot flow. Replaces the generated-header approach of
// fpga/sw/openeye_test.c: new models need no firmware rebuild.

#include <stdio.h>
#include <stdlib.h>

#include "platform.h"
#include "xparameters.h"

#include "openeye_runtime.h"

int hal_xaxidma_init(openeye_hal *hal, u16 device_id);

#ifndef OPENEYE_ARTIFACT_ADDR
#define OPENEYE_ARTIFACT_ADDR      0x10000000u
#endif
#ifndef OPENEYE_ARTIFACT_LEN_ADDR
#define OPENEYE_ARTIFACT_LEN_ADDR  0x0FFFFFF8u
#endif

int main(void) {
    init_platform();
    setvbuf(stdout, NULL, _IONBF, 0);

    uint8_t *artifact = (uint8_t *)OPENEYE_ARTIFACT_ADDR;
    uint32_t artifact_len = *(volatile uint32_t *)OPENEYE_ARTIFACT_LEN_ADDR;

    openeye_model model;
    int rc = openeye_model_parse(artifact, artifact_len, 0, &model);
    if (rc != OPENEYE_OK) {
        printf("artifact parse failed: %d\n", rc);
        return rc;
    }
    printf("model: %lu transfers, output %lu bytes\n",
           (unsigned long)model.n_transfers, (unsigned long)model.output_len);

    openeye_hal hal;
    if (hal_xaxidma_init(&hal, XPAR_AXIDMA_0_DEVICE_ID)) {
        puts("DMA init failed");
        return -1;
    }

    uint8_t *output = malloc(model.output_len);
    if (!output) return -1;

    if (model.reference) {
        rc = openeye_selftest(&model, &hal, output, model.output_len);
        printf(rc == OPENEYE_OK ? "selftest OK\n"
                                : "selftest FAILED (%d)\n", rc);
    } else {
        rc = openeye_run(&model, &hal, NULL, 0, output, model.output_len);
        printf("run: %d\n", rc);
    }

    free(output);
    cleanup_platform();
    return rc;
}
