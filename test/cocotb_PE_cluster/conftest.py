"""PE cluster regression size and waveform controls."""
from .conv_cases import FIELDS, convolution_cases
import pytest


def pytest_addoption(parser):
    group = parser.getgroup("OpenEye PE cluster")
    group.addoption("--cluster-regression", choices=("smoke", "matrix", "extended"),
                    default="smoke", help="Convolution selection (known reproducers always included).")
    group.addoption("--cluster-waves", action="store_true", default=False,
                    help="Write a waveform in each PE cluster case's simulation directory.")


def pytest_generate_tests(metafunc):
    if metafunc.function.__name__ == "test_pe_cluster_conv":
        cases = convolution_cases(metafunc.config.getoption("--cluster-regression"))
        metafunc.parametrize(FIELDS, cases, ids=["-".join(map(str, case)) for case in cases])


@pytest.fixture(autouse=True)
def cluster_waveforms(request, monkeypatch):
    monkeypatch.setenv("OPENEYE_CLUSTER_WAVES",
                       "1" if request.config.getoption("--cluster-waves") else "0")
