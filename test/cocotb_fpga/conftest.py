def pytest_addoption(parser):
    parser.addoption(
        "--target-dir",
        action="store",
        default=None,
        help="Optionales Zielverzeichnis für Waveforms (.fst) und Logs (Build bleibt lokal in .temp)"
    )