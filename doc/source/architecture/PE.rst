
########################
Processing Element (PE)
########################

The PE is implemented in PE.v.

**************
Architecture
**************

The PE contains the following components:

* Two SPads for the `iacts`:

  * one SPad for the addresses (by default, this has a depth of 9 words of 4 bit length)
  * one SPad for the values (by default, this has a depth of 16 words of 12 bit length, consisting of 8 bit data and 4 bit overhead)

* Two SPads for the `wghts`:

  * one SPad for the addresses (by default, this has a depth of 16 words of 7 bit length)
  * one SPad for the values (by default, this has a depth of 96 words of 24 bit length, consisting of two tuples with 8 bit data and 4 bit overhead each)

* One SPad for the `psums`:

  * A true dual port memory of size 32 words with 20 bits each

* Two :math:`8\times 8` bit multipliers
* Two :math:`20\times 20` bit adders
* Two data handlers to transfer a parallel to serial conversion
* Multiple muxes for routing
* A state machine for control 

**************
Operation
**************

The operation of the PE is performed in three steps:

1. Get the data (`iact_data`, `wght_data`) and store it in the correspondig `SPads`
2. Compute the partial sums and store it in the PSum `SPad`
3. Get the bias, add it to PSUM and write the result output port

**************
Example 1
**************
Assuming we have a filter kernel :math:`\mathbf{W}` with the dimensions :math:`6\times 6\times 12` (i.e. rows :math:`r=6`, columns :math:`s=6` and filters :math:`f=12`). In this case, a single
PE just uses the first row of the kernel, i.e., we just look at the part
:math:`1\times 6 \times 12`. The other rows are processed in the other PEs of a PE cluster.

Hence, in this example, :math:`\mathbf{W}` might look like

.. math::

    \mathbf{W}_{1,1:12,1:6}=
    \begin{bmatrix}
    1  &  2  & 3  & 4  & 5  & 6  & 7  & 8  & 9  & 10 & 11 & 12 \\
    13 & 14 & 15 & 16 & 17 & 18 & 19 & 20 & 21 & 22 & 23 & 24 \\
    25 & 26 & 27 & 28 & 29 & 30 & 31 & 32 & 33 & 34 & 35 & 36 \\
    37 & 38 & 39 & 40 & 41 & 42 & 43 & 44 & 45 & 46 & 47 & 48 \\
    49 & 50 & 51 & 52 & 53 & 54 & 55 & 56 & 57 & 58 & 59 & 60 \\
    61 & 62 & 63 & 64 & 65 & 66 & 67 & 68 & 69 & 70 & 71 & 72 \\
    \end{bmatrix}

Furthermore, we have input data :math:`\mathbf{I}` of the dimension height :math:`h=6`, width :math:`w=6` and channels :math:`c=1`.
In this case, the PE just uses the first row of the input data, i.e., we just look at the part :math:`1\times 6 \times 1`.

This might look like this:

.. math::

    \mathbf{I}_{1,1:6,1}=
    \begin{bmatrix}
        1  & 2  & 3  & 4  & 5  & 6
    \end{bmatrix}

Furthermore, we need the bias matrix. Since the dimension of the filter is :math:`1\times 6 \times 12` and the dimension of the input data is :math:`1\times 6 \times 1`, the dimension of the bias matrix is :math:`1\times 12 \times 1`.

For example, this might look like 

.. math::
    \mathbf{B}_{1,1:12,1}=
    \begin{bmatrix}
    1  &  2  & 3  & 4  & 5  & 6  & 7  & 8  & 9  & 10 & 11 & 12 \\
    \end{bmatrix}


Then, the PE computes 

.. math::
    \mathbf{O} =     


Hence, the result should be

.. math::
    \mathbf{O} =  
    \begin{bmatrix}
        862 & 884 & 906 & 928 & 950 & 972 & 994 & 1016 & 1038 & 1082 & 1104 \\
    \end{bmatrix}

**************
Reading in the data
**************

The data is written to the PE using the port `iact_data_i` when `iact_en_i='1'`. In each clock cycle, `iact_data_i` transfers 24 bits of data in the default configuration. Input data is written to the PE in the form of packets, which have the following content:

1. "Adresses" of the input data.
    In this case, the `iact_data_i` is interpreted as a block of 6 *end* adresses of input data rows (each adress has a width of 4 bits) that should be stored consecutively in the `iact` address SPad. 
    In total, we need 6 clock cycles to store these addresses in the `iact_addr_SPad`.   During this time, all further inputs are ignored.
    
    Afterwards, we transfer the remaining 3 addresses (i.e., 12 bits). The remaining 12 bits are ignored. We need 3 clock cycles again to store the adresses in the SPad.

    Example:
        For the example data above, the contents of `iact_data_i` in the first clock cycle is `[0, 0, 0, 0, 0, 6]`. Then we wait for 5 cycles.
        Subsequently, it contains `[X, X, X, 0, 0, 0]` and we wait for 2 additional clock cycles.

2. Now, we start to transfer the actual data. In this case, we transfer two samples at once and need two clock cycles to store the data in the `iact` data SPad. The 24 bits contain two payload data items of 8 bits each and two 4 bit offsets (which are only relevant if we have sparse data, see the example below).

    Example:
        For the example data above, the contents of `iact_data_i` is
        `[0, 2, 0, 1]`, where the first `0` requires 4 bits, the payload part containing the `2` requires 8 bits. The same is true for  `[..., 0, 1]`. Hence, the payload transfer is `[0, 2, 0, 1], [0, 4, 0, 3], [0, 6, 0 5]`.

3. At the same time (i.e. in parallel to the `iact` transfer), we transfer the weight addresses and subsequently the actual weights.

    The weight addresses are transferred in triples (i.e., three addresses per cycle) of 24 bits. Each address requires 7 bits. Hence, the three MSBs are ignored, the remaining 21 bits contain the actual addresses. 
    Hence, for the example above, the transfer is `[XXX, 18, 12, 6]`, where `XXX` denote the three ignored bits. Subsequently we have to wait two cycles to store all addresses in the weight SPad.
    In the following cycles, we transfer `[XXX, 36, 30, 24]` and pad the transfer with `[XXX, 0, 0, 0]`.

    It is important to note that the weight SPad contains two weights per word. Therefore, the addresses have only half of the number of columns of the weight matrix (i.e., 6 instead of 12 for the example data above).

4. Furthermore, we transfer the actual weights. Since the SPad has a width of 24 bit and the data input stream has a width of 24 bits, we can transfer one weight per cycle. Because SIMD is used, every word of the weight data SPad contains 2 weights.

**************
Computing
************** 

The computation is controlled by the internal FSM of the PE.
This was in `idle` state during the data gathering procedure described above. 

1. The computation starts when the `compute_i` input is asserted. The FSM transitions to the data loading phase.
2. During the data loading phase, we load the first entry from the `Iact_addr_SPad`, which specifies how many `iacts` are stored in the first row. For the example above, we obtain the number 6. This means that we can load 6 values from the `Iact_addr_SPad`.
3. Subsequently, we load the data from `Iact_data_SPad`, which contains the actual `iact` and the overhead, which specifies the offset for the weight. Since the example above is not sparse, we have an offset of 0. Hence, we load the weight address at address `0`.
4.  We load the corresponding weight address from the `Wght_addr_SPad`, which specifies the the position in `Wght_data_SPad`. 
5.  The value from `Wght_data_SPad` contains the actual weight and the overhead, which indicates the address of the `Psum_SPad`, where the result of the computation should be stored.
6. If the maximum limit of the `Wght_addr_SPad` is about to be reached, we want to continue with the next `iact_data` value. Hence, we have to increment the address that goes into `Iact_data_SPad`.
7. If the maximum limit of the `Iact_addr_SPad` is about to be reached, we are done with the computation for now. Hence, we have to wait until the computation for the current row of the `iacts` is finished.
8. Now, we can output the data. 

**************
Example 2
**************

In this example, we look at multiple input channels.

Assuming we have a filter kernel :math:`\mathbf{W}` with the dimensions :math:`4\times 4\times 12` (i.e. rows :math:`r=4`, columns :math:`s=4` and filters :math:`f=12`). In this case, a single
PE just uses the first row of the kernel, i.e., we just look at the part
:math:`1\times 6 \times 12`. The other rows are processed in the other PEs of a PE cluster.

Hence, in this example, :math:`\mathbf{W}` might look like

.. math::
    \mathbf{W}_{1,1:12,1:6}=
    \begin{bmatrix}
    1 & 2 & 3 & 4 & 5 & 6 & 7 & 8 & 9 & 10 & 11 & 12 \\
    13 & 14 & 15 & 16 & 17 & 18 & 19 & 20 & 21 & 22 & 23 & 24 \\
    25 & 26 & 27 & 28 & 29 & 30 & 31 & 32 & 33 & 34 & 35 & 36 \\
    37 & 38 & 39 & 40 & 41 & 42 & 43 & 44 & 45 & 46 & 47 & 48 \\
    49 & 50 & 51 & 52 & 53 & 54 & 55 & 56 & 57 & 58 & 59 & 60 \\
    61 & 62 & 63 & 64 & 65 & 66 & 67 & 68 & 69 & 70 & 71 & 72 \\
    73 & 74 & 75 & 76 & 77 & 78 & 79 & 80 & 81 & 82 & 83 & 84 \\
    85 & 86 & 87 & 88 & 89 & 90 & 91 & 92 & 93 & 94 & 95 & 96 \\
    \end{bmatrix}

Furthermore, we have input data :math:`\mathbf{I}` of the dimension height :math:`h=4`, width :math:`w=4` and channels :math:`c=2`.
In this case, the PE just uses the first row of the input data, i.e., we just look at the part :math:`1\times 4 \times 2`.

This might look like this:

.. math::
    \mathbf{I}_{1,1:4,1:2}=
    \begin{bmatrix}
        1 & 2 & 3 & 4 \\
        5 & 6 & 7 & 8 \\
    \end{bmatrix}

Furthermore, we need the bias matrix. Since the dimension of the filter is :math:`1\times 6 \times 12` and the dimension of the input data is :math:`1\times 6 \times 1`, the dimension of the bias matrix is :math:`1\times 12 \times 1`.

For example, this might look like 

.. math::
    \mathbf{B}_{1,1:12,1}=
    \begin{bmatrix}
    1  &  2  & 3  & 4  & 5  & 6  & 7  & 8  & 9  & 10 & 11 & 12 \\
    \end{bmatrix}

Then, the PE computes 

.. math::
    \mathbf{O} =     

Hence, the result should be

.. math::
    \mathbf{O} =  
    \begin{bmatrix}
        2053 & 2090 & 2127 & 2164 & 2201 & 2238 & 2275 & 2312 & 2349 & 2386 & 2423 & 2460
    \end{bmatrix}

...

**************
Example 3
**************

In this example, we consider sparsity and use the same dimensions for :math:`\mathbf{I}` and :math:`\mathbf{W}`.

In this example, :math:`\mathbf{W}` might look like

.. math::
    \mathbf{W}_{1,1:12,1:6}=
    \begin{bmatrix}
    1 & 0 & 3 & 0 & 0 & 6 & 7 & 0 & 0 & 10 & 0 & 12 \\
    0 & 14 & 0 & 0 & 17 & 0 & 0 & 0 & 20 & 22 & 0 & 24 \\
    0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
    37 & 0 & 39 & 0 & 41 & 0 & 43 & 0 & 45 & 0 & 47 & 0 \\
    0 & 50 & 0 & 52 & 0 & 54 & 0 & 56 & 0 & 58 & 0 & 60 \\
    0 & 0 & 0 & 0 & 0 & 0 & 0 & 68 & 69 & 70 & 71 & 72 \\
    0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
    0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
    \end{bmatrix}

Furthermore, :math:`\mathbf{I}` looks like 

.. math::
    \mathbf{I}_{1,1:4,1:2}=
    \begin{bmatrix}
        1 & 0 & 3 & 0 \\
        0 & 6 & 0 & 8 \\
    \end{bmatrix}
