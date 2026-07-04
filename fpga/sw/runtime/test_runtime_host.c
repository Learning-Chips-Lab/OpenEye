// This file is part of the OpenEye project.
// © Fachhochschule Dortmund – University of Applied Sciences and Arts (until 2025), Universität Duisburg-Essen (since 2025).
// SPDX-License-Identifier: SHL-2.1
// For more details, see the LICENSE file in the root directory of this project.

// Host-side end-to-end test of the artifact runtime, no hardware needed:
// loads an .oeye file, replays it through the mock HAL (readback served
// from the embedded reference) and checks parsing, transfer accounting
// and the self-test path.
//
//   ./test_runtime_host model.oeye

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "openeye_runtime.h"
#include "hal_mock.h"

static uint8_t *read_file(const char *path, uint32_t *len_out) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *buf = malloc(len);
    if (!buf || fread(buf, 1, len, f) != (size_t)len) {
        fclose(f);
        free(buf);
        return NULL;
    }
    fclose(f);
    *len_out = (uint32_t)len;
    return buf;
}

#define CHECK(cond, msg)                                        \
    do {                                                        \
        if (!(cond)) {                                          \
            fprintf(stderr, "FAIL: %s (%s:%d)\n", msg,          \
                    __FILE__, __LINE__);                        \
            return 1;                                           \
        }                                                       \
    } while (0)

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <model.oeye>\n", argv[0]);
        return 2;
    }

    uint32_t len = 0;
    uint8_t *data = read_file(argv[1], &len);
    CHECK(data, "read artifact file");

    /* Corrupt-input robustness before the happy path. */
    openeye_model m;
    CHECK(openeye_model_parse(data, 8, 0, &m) == OPENEYE_ERR_TRUNCATED,
          "truncated file rejected");
    data[0] ^= 0xFF;
    CHECK(openeye_model_parse(data, len, 0, &m) == OPENEYE_ERR_MAGIC,
          "bad magic rejected");
    data[0] ^= 0xFF;

    CHECK(openeye_model_parse(data, len, 0, &m) == OPENEYE_OK, "parse");
    printf("parsed: %u transfers, output %u B, reference %u B\n",
           m.n_transfers, m.output_len, m.reference_len);
    CHECK(m.n_transfers > 0, "has transfers");

    uint32_t total_blob_bytes = 0;
    for (uint32_t i = 0; i < m.n_transfers; i++) {
        openeye_transfer t;
        CHECK(openeye_model_transfer(&m, i, &t) == OPENEYE_OK, "transfer");
        CHECK(t.length % 8 == 0, "8-byte aligned blob");
        total_blob_bytes += t.length;
    }

    /* Hardware hash mismatch must be rejected (if artifact carries one). */
    if (m.hw_hash) {
        CHECK(openeye_model_parse(data, len, m.hw_hash ^ 1, &m) ==
              OPENEYE_ERR_HWHASH, "hw hash mismatch rejected");
        CHECK(openeye_model_parse(data, len, m.hw_hash, &m) == OPENEYE_OK,
              "hw hash match accepted");
    }

    openeye_hal hal;
    hal_mock_state mock;
    hal_mock_init(&hal, &mock, m.reference, m.reference_len);

    uint8_t *output = malloc(m.output_len ? m.output_len : 1);
    CHECK(output, "alloc output");

    if (m.reference) {
        CHECK(openeye_selftest(&m, &hal, output, m.output_len) == OPENEYE_OK,
              "selftest passes against embedded reference");
        CHECK(mock.n_writes == m.n_transfers, "all transfers written");
        CHECK(mock.bytes_written == total_blob_bytes, "all bytes written");
        CHECK(mock.n_reads >= 1, "readback happened");

        /* A corrupted response must fail the self-test. */
        uint8_t *bad_ref = malloc(m.reference_len);
        CHECK(bad_ref, "alloc bad ref");
        memcpy(bad_ref, m.reference, m.reference_len);
        bad_ref[0] ^= 0xFF;
        hal_mock_init(&hal, &mock, bad_ref, m.reference_len);
        CHECK(openeye_selftest(&m, &hal, output, m.output_len) ==
              OPENEYE_ERR_SELFTEST, "corrupted output detected");
        free(bad_ref);
    } else {
        hal_mock_init(&hal, &mock, NULL, 0);
        CHECK(openeye_run(&m, &hal, NULL, 0, output, m.output_len) ==
              OPENEYE_OK, "run without reference");
    }

    /* Input patching: only valid if the first transfer has a window. */
    openeye_transfer t0;
    openeye_model_transfer(&m, 0, &t0);
    hal_mock_init(&hal, &mock, m.reference, m.reference_len);
    if (t0.flags & OPENEYE_FLAG_HAS_INPUT_PATCH) {
        uint8_t *input = calloc(1, t0.patch_len);
        CHECK(input, "alloc input");
        memset(input, 0xAB, t0.patch_len);
        CHECK(openeye_run(&m, &hal, input, t0.patch_len, output,
                          m.output_len) == OPENEYE_OK, "run with input");
        CHECK(memcmp(t0.blob + t0.patch_offset, input, t0.patch_len) == 0,
              "input patched into blob");
        free(input);
    } else {
        uint8_t dummy = 0;
        CHECK(openeye_run(&m, &hal, &dummy, 1, output, m.output_len) ==
              OPENEYE_ERR_NO_PATCH, "patch without window rejected");
    }

    free(output);
    free(data);
    puts("all runtime tests passed");
    return 0;
}
