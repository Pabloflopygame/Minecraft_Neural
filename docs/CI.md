# Continuous integration and code health

The GitHub Actions workflow runs on branch pushes and pull requests. Its weekly
schedule (Mondays 06:00 UTC) rechecks vulnerabilities even without code changes.

| Job | Scope |
| --- | --- |
| dataset (Ubuntu and Windows) | Offline dataset and installer checks |
| cpu | Functional and regression tests, checkpoints, deterministic CPU runs, mid-training cancellation, coverage, and CPU performance |
| quality | Ruff correctness checks across all source; broader lint for newly added tests |
| security | pip-audit of the installed environment including PyTorch CPU |

## Local execution

Install PyTorch and the runtime requirements first, then install
requirements-dev.txt. From the repository root:

    python -m pytest tests -q -m "not performance" --cov=minecraft_biomas --cov=scripts --cov=setup --cov-report=term-missing --cov-fail-under=30
    python -m pytest tests/test_performance.py -q -m performance
    ruff check .
    ruff check --select E4,E7,E9,F tests/test_ci_regressions.py tests/test_performance.py
    python -m pip install pip-audit
    python -m pip_audit --local

## Interpretation

- The initial project-wide coverage minimum is 30%; raise it as more code
  becomes covered. Coverage cannot guarantee that tests have good assertions.
- Ruff starts with critical correctness checks on legacy code to avoid an
  unrelated mass cleanup. New test files receive broader E4/E7/E9/F checks.
- CPU performance checks have very generous thresholds because shared runner
  performance varies. They detect drastic regressions, not small speed changes.
- Reproducibility is tested with the same seed, software, CPU, and input images.
  Bit-identical results across all hardware/OS versions are not guaranteed.
- The security audit checks the installed dependency versions and needs internet
  access to query advisories. Newly published advisories may make later runs fail.
- These tests are offline and synthetic except for dependency installation and
  querying vulnerability information.
