"""Filesystem layout of the repository: where inputs live and where outputs go.

    <repo>/data/      inputs  (SPECFEM reference data, tracked in git)
    <repo>/results/   outputs (summaries, models, cache, logs, plots). Local
                      except the paper figures and the reference runs that
                      .gitignore explicitly un-ignores.

The repository root is located relative to this file, which is correct for the
editable install created by `uv sync`. Both roots can be overridden with the
QUFWI_DATA_DIR and QUFWI_RESULTS_DIR environment variables, for example to
write results to a scratch filesystem on a cluster.
"""

import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.abspath(os.environ.get("QUFWI_DATA_DIR", os.path.join(REPO_ROOT, "data")))
RESULTS_DIR = os.path.abspath(os.environ.get("QUFWI_RESULTS_DIR", os.path.join(REPO_ROOT, "results")))


def data_path(*parts):
    "Path under the data directory, e.g. data_path('anomaly', 'specfem')"
    return os.path.join(DATA_DIR, *parts)


def results_path(*parts):
    "Path under the results directory, e.g. results_path('anomaly', 'cache')"
    return os.path.join(RESULTS_DIR, *parts)
