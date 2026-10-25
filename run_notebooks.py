"""Run the learning notebooks in order, each in its own Python kernel."""

import argparse
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

from jupyter_client import AsyncKernelManager
from jupyter_client.kernelspec import KernelSpecManager
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent
NOTEBOOKS = ["01_data.ipynb", "02_model.ipynb", "03_capacity.ipynb", "04_review.ipynb"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", nargs="?", choices=NOTEBOOKS)
    args = parser.parse_args()
    chosen = [args.notebook] if args.notebook else NOTEBOOKS
    executed = ROOT / "output" / "executed"
    executed.mkdir(parents=True, exist_ok=True)
    runtime = ROOT / "output" / ".runtime"
    runtime.mkdir(exist_ok=True)
    os.environ.setdefault("JUPYTER_RUNTIME_DIR", str(runtime))
    os.environ.setdefault("IPYTHONDIR", str(runtime / "ipython"))
    os.environ.setdefault("MPLCONFIGDIR", str(runtime / "matplotlib"))
    # Use this interpreter without a permanent user-level kernel installation.
    with TemporaryDirectory(prefix="mobility-kernel-", dir=runtime) as temp:
        spec = Path(temp) / "mobility"
        spec.mkdir()
        (spec / "kernel.json").write_text(
            json.dumps(
                {
                    "argv": [
                        sys.executable,
                        "-m",
                        "ipykernel_launcher",
                        "-f",
                        "{connection_file}",
                    ],
                    "display_name": "MobilityDemand",
                    "language": "python",
                    "env": {"OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"},
                }
            ),
            encoding="utf-8",
        )
        manager = KernelSpecManager(kernel_dirs=[temp], ensure_native_kernel=False)
        for name in chosen:
            print(f"Running {name}", flush=True)
            notebook = nbformat.read(ROOT / "notebooks" / name, as_version=4)
            client = NotebookClient(
                notebook,
                timeout=300,
                kernel_name="mobility",
                resources={"metadata": {"path": str(ROOT / "notebooks")}},
            )
            client.km = AsyncKernelManager(
                kernel_name="mobility", kernel_spec_manager=manager
            )
            client.owns_km = True
            try:
                client.execute()
            except Exception:
                nbformat.write(notebook, executed / name)
                raise
            nbformat.write(notebook, executed / name)
            print(f"Finished {name}", flush=True)


if __name__ == "__main__":
    main()
