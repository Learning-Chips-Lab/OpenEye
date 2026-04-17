.. _fpga_simulation:

Waveform / Simulation Facilities
==================================

``fst_path`` and ``$dumpvars``
--------------------------------

.. code-block:: verilog

   `ifndef NO_TRACE
     initial begin
       if ($value$plusargs("FST_PATH=%s", fst_path)) begin
         $dumpfile(fst_path);
         $dumpvars(0, OpenEye_FPGA);
       end else begin
         $dumpfile("OpenEye_FPGA.fst");
         $dumpvars(0, OpenEye_FPGA);
       end
     end
   `endif

When ``NO_TRACE`` is not defined, the simulation dumps all signals in ``OpenEye_FPGA`` to
an FST file. The path can be overridden with ``+FST_PATH=<path>`` on the simulator command
line.

``UNPACKED_TRACES_ENABLED``
-----------------------------

When set to ``1`` (default), generates named wire aliases for all arrays that would
otherwise be invisible in VCD/FST waveform viewers:

- ``pooling_stage_1_traces[0..7]``
- ``pooling_stage_2_traces[0..3]``
- ``pooling_stage_3_traces[0..1]``
- ``pooling_stage_out_trace[0..31]`` (``pooling_regs``)
- ``quantized_out_trace[0..7]`` (``quantized_value_reg``)

Debug Ports
-----------

As listed in :ref:`fpga_ports`, several debug output ports expose internal SRAM control
signals for external ChipScope/SignalTap logic analyzers on FPGA.
