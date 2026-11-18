.. _test:

Tests and Verification
======================

The launcher ``test/cocotb_PE/run_recorded.sh`` runs selected Cocotb
tests locally and in GitLab CI. Icarus is the default simulator.
The GitLab pipeline also runs ``pe_seed42`` with Verilator in Docker.
Each simulation run writes its results to a separate directory.

Simulation cases
----------------

.. list-table:: Cases included in the CI workflow
   :header-rows: 1
   :widths: 45 30 25

   * - Case
     - Testbench
     - Seed
   * - ``pe_seed42``
     - PE
     - 42
   * - ``pe_seed43``
     - PE
     - 43
   * - ``pe_simple_seed42``
     - PE_simple
     - 42

Local execution
---------------

From the repository root, activate the environment containing
the simulation dependencies:

.. code-block:: bash

   source .venv/bin/activate
   export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

Run a case and inspect its exit status:

.. code-block:: bash

   bash test/cocotb_PE/run_recorded.sh pe_seed42
   printf 'Exit status: %s\n' "$?"

The other cases are invoked with:

.. code-block:: bash

   bash test/cocotb_PE/run_recorded.sh pe_seed43
   bash test/cocotb_PE/run_recorded.sh pe_simple_seed42

Run cases sequentially because generated parameter and input
files may be shared.

Verilator simulation
--------------------

Use Cocotb 2.1.0 and Verilator 5.036 or newer. With a compatible
``verilator`` executable on ``PATH``, run:

.. code-block:: bash

   SIM=verilator bash test/cocotb_PE/run_recorded.sh pe_seed42

To select another installation for one run, prefix the command with
``PATH="/path/to/verilator/bin:$PATH"``, using its actual bin directory.
Verilator waveform tracing is disabled by default.

Docker simulation
-----------------

Docker supplies the Python environment and simulators.
With Docker running, build the image from the repository root:

.. code-block:: bash

   docker build --progress=plain -t openeye-sim:verilator-5.036 .

Run ``pe_seed42`` with Verilator inside the container:

.. code-block:: bash

   docker run --rm \
       --user "$(id -u):$(id -g)" \
       --mount "type=bind,source=$PWD,target=/workspace" \
       --env SIM=verilator \
       openeye-sim:verilator-5.036
   printf 'Exit status: %s\n' "$?"

The mounted checkout preserves results in
``test/cocotb_PE/results/`` after the container exits.


Recorded outputs
----------------

Each run prints its unique results directory:

* PE: ``test/cocotb_PE/results/``
* PE_simple: ``test/cocotb_PE_simple/results/``

The recorded outputs include:

* ``metadata.txt``: case, seed, simulator, source revision, tool version,
  working directory, command, local Git status and exit status.
* ``output.log``: build output and simulation summary.
* ``results.xml``: Cocotb test results, when generated.
* ``source.patch``: changes to tracked files relative to the recorded revision.
* ``PE.fst`` or ``PE_simple.fst``: waveforms, when generated.
* ``sim_build/``: simulator build files.

Untracked files are listed in the metadata but their contents are
not included in ``source.patch``.

Check the exit status and Cocotb summary. Failed or interrupted runs
may have incomplete outputs.


GitLab CI
---------

The root ``.gitlab-ci.yml`` includes the hardware jobs from
``.gitlab/ci/``. The pipeline is configured to run on pushes and
merge requests.

* ``pe`` runs PE seeds 42 and 43 and PE_simple seed 42 with Icarus.
  Its artifact is ``pe-results``.
* ``pe-verilator`` builds the Docker image and runs PE seed 42 with
  Verilator. Its artifact is ``verilator-pe-results``. The file
  ``container-image.txt`` records the ID of the image used.
* ``lint`` runs the ``pe_dense`` and ``parallel`` profiles as separate
  jobs. Their artifacts are ``lint-pe_dense`` and ``lint-parallel``.
* ``pe-dense`` runs dense-PE synthesis with Yosys. Its artifact is
  ``synthesis-pe-dense``.

In GitLab, open **Build > Pipelines**, select the pipeline for your
commit, then open a job to read its log or download its artifacts.
Each job attempts to upload available results after an ordinary
failure. If setup fails early, there may be no files to upload.


Extending the framework
-----------------------

Additional PE seeds can be supplied as numeric arguments. For example,
to run PE with seed 44:

.. code-block:: bash

   bash test/cocotb_PE/run_recorded.sh 44

To include another case in CI:

1. Run the case locally.
2. In ``.gitlab/ci/pe-simulation.yml``, add its launcher command
   alongside the existing cases, using the same timeout and
   ``|| status=$?`` failure handling.
3. If the case writes to a new results directory, add that directory
   to ``artifacts:paths``.
4. Update the simulation-case table above.

The existing artifact paths already cover additional PE seeds.

To support another testbench, extend the launcher's case selection,
testbench invocation and output handling. Add the corresponding CI
command and artifact path, then verify both successful execution
and failure reporting.

Recorded lint checks
--------------------

Run lint from the repository root:

.. code-block:: bash

   bash test/lint/run_recorded.sh pe_dense
   bash test/lint/run_recorded.sh parallel

Use ``pe_dense`` for the dense PE configuration or ``parallel`` for
``OpenEye_Parallel`` with its default parameters.

Each run saves results in a new folder under ``test/lint/results/``:

* ``output.log``: warnings and errors.
* ``metadata.txt``: configuration, Git revision, tool version and exit status.
* ``source.patch``: local changes to tracked files.

A nonzero exit status means the lint check failed.

Recorded synthesis
------------------

Run dense-PE synthesis from the repository root:

.. code-block:: bash

   bash yosys/run_recorded.sh pe_dense

The launcher uses ``yosys/pe_dense.tcl`` to synthesize the dense PE for
GateMate. Yosys checks the synthesized design for structural problems.
The launcher then checks that ``resources.json`` and ``netlist.json``
are valid JSON.

Each run saves the resource counts (``resources.json``), synthesized
design (``netlist.json``) and Yosys log (``output.log``) in a new
directory under ``yosys/results/``. The flow stops before placement,
routing and timing analysis.