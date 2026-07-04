// This file is part of the OpenEye project.
// © Fachhochschule Dortmund – University of Applied Sciences and Arts (until 2025), Universität Duisburg-Essen (since 2025).
// SPDX-License-Identifier: SHL-2.1
// For more details, see the LICENSE file in the root directory of this project.

#include "openeye_runtime.h"

/* Header field offsets, see artifact.py for the authoritative layout. */
#define H_MAGIC        0
#define H_VERSION      4
#define H_HDRSIZE      6
#define H_HWHASH       8
#define H_NTRANSFERS  12
#define H_TABLE_OFF   16
#define H_BLOB_OFF    20
#define H_BLOB_LEN    24
#define H_INPUT_LEN   28
#define H_OUTPUT_LEN  32
#define H_REF_OFF     36
#define H_REF_LEN     40
#define H_IN_SCALE    44
#define H_IN_ZP       48
#define H_OUT_SCALE   52
#define H_OUT_ZP      56
#define HEADER_SIZE   64

#define T_BLOB_OFF     0
#define T_LENGTH       4
#define T_OUT_BYTES    8
#define T_FLAGS       12
#define T_PATCH_OFF   16
#define T_PATCH_LEN   20
#define T_LAYER_ID    24
#define T_REPETITION  26
#define TRANSFER_RECORD_SIZE 32

/* Unaligned little-endian loads: the artifact may sit at any address. */
static uint16_t ld16(const uint8_t *p) {
    return (uint16_t)(p[0] | (p[1] << 8));
}

static uint32_t ld32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static float ldf32(const uint8_t *p) {
    union { uint32_t u; float f; } v;
    v.u = ld32(p);
    return v.f;
}

uint32_t openeye_fnv1a32(const uint8_t *data, uint32_t len) {
    uint32_t h = 0x811C9DC5u;
    for (uint32_t i = 0; i < len; i++) {
        h ^= data[i];
        h *= 0x01000193u;
    }
    return h;
}

int openeye_model_parse(const uint8_t *data, uint32_t len,
                        uint32_t expected_hw_hash, openeye_model *m) {
    if (!data || !m || len < HEADER_SIZE) return OPENEYE_ERR_TRUNCATED;
    if (ld32(data + H_MAGIC) != OPENEYE_MAGIC) return OPENEYE_ERR_MAGIC;
    if (ld16(data + H_VERSION) != OPENEYE_FMT_VERSION) return OPENEYE_ERR_VERSION;
    if (ld16(data + H_HDRSIZE) != HEADER_SIZE) return OPENEYE_ERR_VERSION;

    m->base = data;
    m->size = len;
    m->hw_hash = ld32(data + H_HWHASH);
    m->n_transfers = ld32(data + H_NTRANSFERS);

    uint32_t table_off = ld32(data + H_TABLE_OFF);
    uint32_t blob_off  = ld32(data + H_BLOB_OFF);
    uint32_t blob_len  = ld32(data + H_BLOB_LEN);
    uint32_t ref_off   = ld32(data + H_REF_OFF);
    uint32_t ref_len   = ld32(data + H_REF_LEN);

    if (m->n_transfers == 0) return OPENEYE_ERR_PARAM;
    /* Overflow-safe bounds checks (all offsets are u32). */
    if (table_off > len ||
        (uint64_t)table_off + (uint64_t)m->n_transfers * TRANSFER_RECORD_SIZE > len)
        return OPENEYE_ERR_TRUNCATED;
    if (blob_off > len || (uint64_t)blob_off + blob_len > len)
        return OPENEYE_ERR_TRUNCATED;
    if (ref_off && (ref_off > len || (uint64_t)ref_off + ref_len > len))
        return OPENEYE_ERR_TRUNCATED;

    if (expected_hw_hash && m->hw_hash && expected_hw_hash != m->hw_hash)
        return OPENEYE_ERR_HWHASH;

    m->table = data + table_off;
    m->blobs = data + blob_off;
    m->input_len  = ld32(data + H_INPUT_LEN);
    m->output_len = ld32(data + H_OUTPUT_LEN);
    m->reference = ref_off ? data + ref_off : (const uint8_t *)0;
    m->reference_len = ref_off ? ref_len : 0;
    m->input_scale  = ldf32(data + H_IN_SCALE);
    m->input_zero_point  = (int32_t)ld32(data + H_IN_ZP);
    m->output_scale = ldf32(data + H_OUT_SCALE);
    m->output_zero_point = (int32_t)ld32(data + H_OUT_ZP);

    /* Validate every transfer record once up front. */
    uint32_t blob_end = blob_len;
    for (uint32_t i = 0; i < m->n_transfers; i++) {
        const uint8_t *r = m->table + i * TRANSFER_RECORD_SIZE;
        uint32_t b_off = ld32(r + T_BLOB_OFF);
        uint32_t b_len = ld32(r + T_LENGTH);
        if (b_len == 0 || (b_len & 7u)) return OPENEYE_ERR_PARAM;
        if (b_off > blob_end || (uint64_t)b_off + b_len > blob_end)
            return OPENEYE_ERR_TRUNCATED;
        if (ld32(r + T_FLAGS) & OPENEYE_FLAG_HAS_INPUT_PATCH) {
            uint32_t p_off = ld32(r + T_PATCH_OFF);
            uint32_t p_len = ld32(r + T_PATCH_LEN);
            if (p_off > b_len || (uint64_t)p_off + p_len > b_len)
                return OPENEYE_ERR_PARAM;
        }
    }
    return OPENEYE_OK;
}

int openeye_model_transfer(const openeye_model *m, uint32_t i,
                           openeye_transfer *t) {
    if (!m || !t || i >= m->n_transfers) return OPENEYE_ERR_PARAM;
    const uint8_t *r = m->table + i * TRANSFER_RECORD_SIZE;
    t->blob = m->blobs + ld32(r + T_BLOB_OFF);
    t->length = ld32(r + T_LENGTH);
    t->out_bytes = ld32(r + T_OUT_BYTES);
    t->flags = ld32(r + T_FLAGS);
    t->patch_offset = ld32(r + T_PATCH_OFF);
    t->patch_len = ld32(r + T_PATCH_LEN);
    t->layer_id = ld16(r + T_LAYER_ID);
    t->repetition = ld16(r + T_REPETITION);
    return OPENEYE_OK;
}

static void copy_bytes(uint8_t *dst, const uint8_t *src, uint32_t len) {
    for (uint32_t i = 0; i < len; i++) dst[i] = src[i];
}

int openeye_run(const openeye_model *m, const openeye_hal *hal,
                const uint8_t *input, uint32_t input_len,
                uint8_t *output, uint32_t output_cap) {
    if (!m || !hal || !hal->dma_write || !hal->dma_read_start ||
        !hal->dma_read_wait)
        return OPENEYE_ERR_PARAM;
    if (m->output_len && (!output || output_cap < m->output_len))
        return OPENEYE_ERR_PARAM;

    if (input) {
        /* Locate the patch window (by convention on the first transfer). */
        openeye_transfer t0;
        int rc = openeye_model_transfer(m, 0, &t0);
        if (rc) return rc;
        if (!(t0.flags & OPENEYE_FLAG_HAS_INPUT_PATCH))
            return OPENEYE_ERR_NO_PATCH;
        if (input_len > t0.patch_len) return OPENEYE_ERR_PARAM;
        /* The artifact buffer must be writable for input patching. */
        copy_bytes((uint8_t *)(t0.blob + t0.patch_offset), input, input_len);
    }

    for (uint32_t i = 0; i < m->n_transfers; i++) {
        openeye_transfer t;
        int rc = openeye_model_transfer(m, i, &t);
        if (rc) return rc;

        if (t.out_bytes) {
            if (t.out_bytes > output_cap) return OPENEYE_ERR_PARAM;
            /* Arm the read before writing: the accelerator streams
             * results while the transfer is still going in. */
            if (hal->dma_read_start(hal->ctx, output, t.out_bytes))
                return OPENEYE_ERR_DMA;
            if (hal->dma_write(hal->ctx, t.blob, t.length))
                return OPENEYE_ERR_DMA;
            if (hal->dma_read_wait(hal->ctx))
                return OPENEYE_ERR_DMA;
        } else {
            if (hal->dma_write(hal->ctx, t.blob, t.length))
                return OPENEYE_ERR_DMA;
        }
    }
    return OPENEYE_OK;
}

int openeye_selftest(const openeye_model *m, const openeye_hal *hal,
                     uint8_t *scratch, uint32_t scratch_cap) {
    if (!m || !m->reference || m->reference_len < m->output_len)
        return OPENEYE_ERR_PARAM;
    int rc = openeye_run(m, hal, (const uint8_t *)0, 0, scratch, scratch_cap);
    if (rc) return rc;
    for (uint32_t i = 0; i < m->output_len; i++) {
        if (scratch[i] != m->reference[i]) return OPENEYE_ERR_SELFTEST;
    }
    return OPENEYE_OK;
}
