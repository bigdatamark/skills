#!/usr/bin/env python
# [MODULE DOC] CLI script that prints the installed version of selected Python distributions.
# Behavior: when run as a script, calls main and exits the process with its return code.
# CLI arguments: zero or more distribution names; when none are given, the default list
#   (boto3, botocore, awscli) is reported.
# Env vars: none.
# Constants: DEFAULT_PACKAGES - the distribution names reported when no arguments are given.
# Top-level executed code: only the main-guard that calls main and exits with its result.
# Outputs: one line per package on stdout; exit code is always 0.
"""Report installed versions of key dependencies.

Uses importlib.metadata.version() (works for every installed package, unlike
the inconsistently-defined `module.__version__`).

Usage:
    python check_versions.py
    python check_versions.py boto3 botocore transformers
"""

import sys
from importlib.metadata import PackageNotFoundError, version

DEFAULT_PACKAGES = ["boto3", "botocore", "awscli"]


def main() -> int:
    # [FUNCTION DOC] Print the installed version of each requested distribution.
    # Behavior: takes the command-line arguments after the script name as the package list,
    #   or DEFAULT_PACKAGES if there are none. For each name, looks up its installed version via
    #   the package metadata API and prints "name==version"; if the distribution is not installed,
    #   prints "name: NOT INSTALLED" instead.
    # Inputs: none (reads the process command-line arguments).
    # Outputs: returns 0 always; writes one line per package to stdout.
    # Exceptions: catches the package-not-found error per package; others propagate.
    # Side effects: reads process arguments; prints to stdout.
    packages = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_PACKAGES
    for pkg in packages:
        try:
            print(f"{pkg}=={version(pkg)}")
        except PackageNotFoundError:
            print(f"{pkg}: NOT INSTALLED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
