// This file is part of the OpenEye project.
// © Fachhochschule Dortmund – University of Applied Sciences and Arts (until 2025), Universität Duisburg-Essen (since 2025).
// SPDX-License-Identifier: SHL-2.1
// For more details, see the LICENSE file in the root directory of this project.

// Portable runtime for OpenEye .oeye model artifacts.
//
// An artifact is produced offline by src/open_eye/artifact.py and holds
// per-transfer DMA blobs plus metadata. This runtime parses the container
// (zero-copy: the model keeps pointers into the artifact buffer) and
// replays the blobs through the HAL.
//
// Format layout must stay in sync with artifact.py (OPENEYE_FMT_VERSION).
//
// Freestanding: no libc dependencies beyond <stdint.h>/<stddef.h>;
// suitable for bare metal and callable from Rust via the C ABI.

#ifndef OPENEYE_RUNTIME_H
#define OPENEYE_RUNTIME_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#include "openeye_hal.h"

#define OPENEYE_MAGIC        0x4559454Fu /* 'OEYE' */
#define OPENEYE_FMT_VERSION  1u

/* Error codes */
#define OPENEYE_OK                 0
#define OPENEYE_ERR_MAGIC         -1
#define OPENEYE_ERR_VERSION       -2
#define OPENEYE_ERR_TRUNCATED     -3
#define OPENEYE_ERR_HWHASH        -4
#define OPENEYE_ERR_PARAM         -5
#define OPENEYE_ERR_DMA           -6
#define OPENEYE_ERR_NO_PATCH      -7
#define OPENEYE_ERR_SELFTEST      -8

typedef struct {
    const uint8_t *blob;      /* points into the artifact buffer */
    uint32_t length;          /* bytes, multiple of 8 */
    uint32_t out_bytes;       /* expected readback bytes, 0 = write only */
    uint32_t flags;
    uint32_t patch_offset;    /* input window inside blob (if flag set) */
    uint32_t patch_len;
    uint16_t layer_id;
    uint16_t repetition;
} openeye_transfer;

#define OPENEYE_FLAG_HAS_INPUT_PATCH 0x1u

typedef struct {
    const uint8_t *base;      /* artifact buffer (writable if patching) */
    uint32_t size;
    uint32_t hw_hash;
    uint32_t n_transfers;
    const uint8_t *table;     /* raw transfer table */
    const uint8_t *blobs;
    uint32_t input_len;       /* raw input bytes (0 = input baked in) */
    uint32_t output_len;      /* bytes produced by final readback */
    const uint8_t *reference; /* embedded expected output, NULL if none */
    uint32_t reference_len;
    float    input_scale;
    int32_t  input_zero_point;
    float    output_scale;
    int32_t  output_zero_point;
} openeye_model;

/* Parse and validate an artifact in memory. Zero-copy: `data` must stay
 * valid for the model's lifetime. expected_hw_hash of 0 skips the check
 * (as does an artifact built with hash 0). */
int openeye_model_parse(const uint8_t *data, uint32_t len,
                        uint32_t expected_hw_hash, openeye_model *m);

/* Decode transfer record i into *t. Returns OPENEYE_ERR_PARAM if out of
 * range or inconsistent with the blob section. */
int openeye_model_transfer(const openeye_model *m, uint32_t i,
                           openeye_transfer *t);

/* Run inference.
 *
 * input/input_len: raw quantized input, copied over the first transfer's
 *   patch window. Pass NULL to run with the baked-in activations (e.g.
 *   for self-test). Patching writes into the artifact buffer, which must
 *   then be in writable memory.
 * output/output_cap: receives the final readback; output_cap must be
 *   >= m->output_len.
 *
 * Intermediate transfers marked with out_bytes > 0 also read into
 * `output` (the wrapper streams psums back mid-model only when the
 * offline mapper scheduled a DDR spill; the last readback wins). */
int openeye_run(const openeye_model *m, const openeye_hal *hal,
                const uint8_t *input, uint32_t input_len,
                uint8_t *output, uint32_t output_cap);

/* Run with baked-in input and compare the readback against the embedded
 * reference. scratch must hold m->output_len bytes. */
int openeye_selftest(const openeye_model *m, const openeye_hal *hal,
                     uint8_t *scratch, uint32_t scratch_cap);

/* FNV-1a 32-bit; must match artifact.py. Use to compute the expected
 * hardware hash from the same canonical parameter string. */
uint32_t openeye_fnv1a32(const uint8_t *data, uint32_t len);

#ifdef __cplusplus
}
#endif

#endif /* OPENEYE_RUNTIME_H */
