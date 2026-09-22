"""Deterministic convolution selections; tuple order preserves existing test IDs."""
from itertools import product

FIELDS = "SPARSITY_EN,PARALLEL_MACS,SEED,SPARSE_WGHT,SPARSE_IACT,WGHTSIZE_X,IACTSIZE_Y,IACTSIZE_X"

# Eight workloads crossed with both MAC counts and both hardware modes.
# Cover each dimension value, odd/even activation lengths, zero-free data,
# activation-only zeros, weight-only zeros, and simultaneous zeros.
# x, channels, filters, activation sparsity, weight sparsity
SMOKE_WORKLOADS = (
    (2, 2, 6, 0, 0),
    (4, 3, 10, 0, 0),
    (3, 3, 8, 10, 0),
    (4, 2, 6, 30, 0),
    (2, 3, 10, 0, 30),
    (3, 2, 8, 0, 60),
    (4, 3, 8, 20, 40),
    (3, 3, 6, 30, 60),
)

# Six sampled failures documented in test_status_handover.md section 4.2,
# plus the original seed-4 reproducer cited by the focused sparsity sweep.
# Keep these as ordinary tests: a known RTL failure must remain visible.
REPRODUCERS = (
    (1, 1, 13, 40, 10, 6, 3, 2),
    (1, 2, 0, 50, 20, 6, 3, 4),
    (1, 2, 1, 40, 0, 6, 2, 3),
    (1, 2, 5, 60, 30, 10, 3, 2),
    (1, 2, 9, 60, 20, 10, 2, 3),
    (1, 2, 10, 50, 0, 8, 3, 2),
    (1, 2, 4, 40, 20, 6, 2, 2),
)


def convolution_cases(level):
    if level == "smoke":
        cases = [(s, m, 0, sw, si, f, y, x)
                 for s, m in product((0, 1), (1, 2))
                 for x, y, f, si, sw in SMOKE_WORKLOADS]
    elif level in ("matrix", "extended"):
        cases = list(product((0, 1), (1, 2),
                             range(16) if level == "extended" else (0,),
                             (0, 10, 20, 30, 40, 50, 60), (0, 10, 20, 30),
                             (10, 8, 6), (3, 2), (4, 3, 2)))
    else:
        raise ValueError(f"Unknown cluster regression level: {level}")
    return list(dict.fromkeys([*cases, *REPRODUCERS]))
