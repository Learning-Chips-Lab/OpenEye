# This file is part of the OpenEye project.
# © Fachhochschule Dortmund – University of Applied Sciences and Arts (until 2025), Universität Duisburg-Essen (since 2025).
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.

# Host-side build of the OpenEye artifact runtime (mock HAL, no hardware).
# The bare-metal build (main_baremetal.c + hal_xaxidma.c) is done in
# Vitis/standalone with the BSP toolchain; just add those files plus
# openeye_runtime.c to the application project.

CC      ?= cc
CFLAGS  ?= -std=c99 -Wall -Wextra -Werror -O2

SRCS = openeye_runtime.c hal_mock.c test_runtime_host.c

test_runtime_host: $(SRCS) openeye_runtime.h openeye_hal.h hal_mock.h
	$(CC) $(CFLAGS) -o $@ $(SRCS)

.PHONY: check clean
check: test_runtime_host
	@test -n "$(ARTIFACT)" || (echo "usage: make check ARTIFACT=model.oeye"; exit 2)
	./test_runtime_host $(ARTIFACT)

clean:
	rm -f test_runtime_host
