#!/usr/bin/env python3
# [MODULE DOC] CLI script that creates (or reuses) an isolated Python virtual environment for
# SageMaker work and installs the skill's requirements file into it.
# Behavior: validates the requested Python version is within 3.10-3.12, chooses uv if it is on
#   PATH (else the standard venv module), reuses an existing env only if its interpreter
#   version matches, otherwise deletes and recreates it, then installs requirements.txt from
#   the skill directory (the parent of this scripts folder) with an upgrade.
# CLI arguments: optional first argument is the venv directory (default ".venv"); optional
#   second argument is the Python major.minor version (default "3.12").
# Env vars: none read directly (PATH is consulted to find uv / python executables).
# Constants: SUPPORTED_MIN and SUPPORTED_MAX - inclusive (major, minor) version bounds;
#   IS_WINDOWS - true when the OS is Windows, used to pick the venv interpreter layout.
# Top-level executed code: the main-guard that calls main and exits with its return code.
# Outputs: progress/errors logged to stderr; exit code 0 on success, 1 on any failure.
"""Create an isolated Python environment for SageMaker work. Cross-platform.

Idempotent. Prefers uv, falls back to the stdlib venv module.

Handles the platform difference in venv layout: the interpreter lives at
`<venv>/bin/python` on macOS/Linux and `<venv>\\Scripts\\python.exe` on Windows.
The path it prints is the correct one for the host OS.

Usage:
    python setup_env.py [VENV_DIR=.venv] [PYTHON_VERSION=3.12]
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

# 3.10-3.12 is the safe zone for modern boto3 + awscli. 3.13+ may work but ML
# libs lag on wheel availability.
SUPPORTED_MIN = (3, 10)
SUPPORTED_MAX = (3, 12)

IS_WINDOWS = os.name == "nt"


def log(msg: str) -> None:
    # [FUNCTION DOC] HELPER: Write a prefixed log message to stderr.
    # Behavior: prints the message prefixed with "[setup_env] " to stderr and flushes.
    # Inputs: msg - text to log.
    # Outputs: none returned; one line on stderr.
    # Exceptions: none.
    # Side effects: writes to stderr.
    print(f"[setup_env] {msg}", file=sys.stderr, flush=True)


def parse_version(v: str) -> tuple[int, int]:
    # [FUNCTION DOC] HELPER: Parse a "major.minor" version string into a pair of integers.
    # Behavior: splits the string on dots, takes the first two parts and converts them to
    #   integers; on a conversion failure logs an error and exits the process with code 1.
    # Inputs: v - version text such as "3.12" (extra parts like a patch number are ignored).
    # Outputs: a (major, minor) integer pair.
    # Exceptions: catches the value conversion error and terminates via a system exit (code 1).
    # Side effects: may log to stderr and exit the process.
    # KNOWN ISSUE: the requested version is later compared to the interpreter's version as text,
    #   so an equivalent spelling such as "3.012" passes validation but never matches, forcing
    #   the env to be deleted and recreated on every run.
    try:
        major, minor = (int(p) for p in v.split(".")[:2])
        return (major, minor)
    except ValueError:
        log(f"ERROR: invalid Python version '{v}' (expected e.g. 3.12)")
        sys.exit(1)


def venv_python(venv_dir: Path) -> Path:
    # [FUNCTION DOC] HELPER: Return the path of the interpreter inside a venv for the current platform.
    # Behavior: on Windows returns venv/Scripts/python.exe, otherwise venv/bin/python.
    # Inputs: venv_dir - path of the virtual environment directory.
    # Outputs: the interpreter path (not checked for existence).
    # Exceptions: none.
    # Side effects: none.
    """Path to the interpreter inside a venv, per platform."""
    if IS_WINDOWS:
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def interpreter_version(python_path: Path) -> str | None:
    # [FUNCTION DOC] HELPER: Report the major.minor version of a given Python interpreter.
    # Behavior: returns nothing if the path does not exist; otherwise runs the interpreter with a
    #   one-line program printing its major.minor version and returns the trimmed stdout if the
    #   process succeeded, or nothing if it failed.
    # Inputs: python_path - path to an interpreter executable.
    # Outputs: a "major.minor" string, or none when missing or not runnable.
    # Exceptions: none caught; an OS error launching the process would propagate.
    # Side effects: spawns a subprocess.
    if not python_path.exists():
        return None
    proc = subprocess.run(
        [str(python_path), "-c",
         "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
        capture_output=True, text=True,
    )
    return proc.stdout.strip() if proc.returncode == 0 else None


def find_base_python(version: str) -> str | None:
    # [FUNCTION DOC] HELPER: Locate a base interpreter of the requested version for creating a venv.
    # Behavior: first looks up "python<version>" on PATH and returns its path if found. Otherwise,
    #   on Windows with the "py" launcher available, runs "py -<version> --version" and, if it
    #   succeeds, returns the sentinel string "py -<version>" which the caller splits into a
    #   command. Returns nothing if neither works.
    # Inputs: version - "major.minor" text.
    # Outputs: an executable path, the "py -X.Y" sentinel, or none.
    # Exceptions: none caught.
    # Side effects: searches PATH; may spawn the py launcher subprocess.
    """Locate a base interpreter for the requested version for `python -m venv`."""
    # Try `pythonX.Y` (Unix) / `pythonX.Y.exe`, then the Windows `py` launcher.
    candidate = shutil.which(f"python{version}")
    if candidate:
        return candidate
    if IS_WINDOWS and shutil.which("py"):
        probe = subprocess.run(["py", f"-{version}", "--version"], capture_output=True, text=True)
        if probe.returncode == 0:
            return f"py -{version}"  # sentinel; expanded by caller
    return None


def main() -> int:
    # [FUNCTION DOC] Create or reuse the venv and install requirements; the script entry point.
    # Behavior: 1) read venv dir (default .venv) and version (default 3.12) from arguments;
    #   2) parse the version and fail if outside SUPPORTED_MIN..SUPPORTED_MAX; 3) pick uv if on
    #   PATH else venv, warning on fallback; 4) compute the venv interpreter path and its current
    #   version; if equal to the requested version text, reuse; otherwise if an env exists with
    #   another version delete the whole directory, then create it via "uv venv --python" or via
    #   the located base interpreter's "-m venv" (failing if no base interpreter is found);
    #   5) locate requirements.txt one directory above this script's folder, fail if absent;
    #   6) install with "uv pip install --python <py> --upgrade -r" or, for venv, first upgrade pip
    #   (result ignored) then "pip install --upgrade -r"; 7) report the interpreter path.
    # Inputs: none (reads the process command-line arguments).
    # Outputs: returns 0 on success, 1 on any failure; logs progress to stderr.
    # Exceptions: none caught; parse_version may exit the process.
    # Side effects: may delete and create directories, run uv/python/pip subprocesses (network
    #   access for package downloads), writes to stderr.
    # KNOWN ISSUE: if the venv directory exists but has no working interpreter (current is none),
    #   it is not removed, and creation into a non-empty/broken directory may fail or be partial.
    # KNOWN ISSUE: the venv fallback never checks that the base interpreter it found is actually
    #   the requested version beyond its name.
    venv_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".venv")
    python_version = sys.argv[2] if len(sys.argv) > 2 else "3.12"

    ver = parse_version(python_version)
    if not (SUPPORTED_MIN <= ver <= SUPPORTED_MAX):
        lo = ".".join(map(str, SUPPORTED_MIN))
        hi = ".".join(map(str, SUPPORTED_MAX))
        log(f"ERROR: Python {python_version} outside supported range {lo}-{hi}")
        log("Use 3.10, 3.11, or 3.12.")
        return 1

    installer = "uv" if shutil.which("uv") else "venv"
    if installer == "venv":
        log("uv not found — falling back to python + venv. Consider: https://docs.astral.sh/uv/")

    py = venv_python(venv_dir)

    # Reuse existing env only if Python version matches.
    current = interpreter_version(py)
    if current == python_version:
        log(f"Env exists at {venv_dir} with Python {current} — reusing")
    else:
        if current is not None:
            log(f"Env at {venv_dir} uses Python {current} (wanted {python_version}) — recreating")
            shutil.rmtree(venv_dir)

        if installer == "uv":
            uv = shutil.which("uv")
            if subprocess.run([uv, "venv", "--python", python_version, str(venv_dir)]).returncode != 0:
                log("ERROR: `uv venv` failed.")
                return 1
        else:
            base = find_base_python(python_version)
            if not base:
                log(f"ERROR: python{python_version} not found on PATH.")
                log("Install via pyenv/asdf/brew/system package manager, the Windows installer, or install uv.")
                return 1
            base_cmd = base.split() if base.startswith("py ") else [base]
            if subprocess.run([*base_cmd, "-m", "venv", str(venv_dir)]).returncode != 0:
                log("ERROR: `python -m venv` failed.")
                return 1
        log(f"Created env at {venv_dir}")

    requirements = Path(__file__).resolve().parent.parent / "requirements.txt"
    if not requirements.is_file():
        log(f"ERROR: requirements.txt not found at {requirements}")
        return 1

    log(f"Installing from {requirements}")
    if installer == "uv":
        uv = shutil.which("uv")
        rc = subprocess.run(
            [uv, "pip", "install", "--python", str(py), "--upgrade", "-r", str(requirements)]
        ).returncode
    else:
        subprocess.run([str(py), "-m", "pip", "install", "--upgrade", "pip"])
        rc = subprocess.run(
            [str(py), "-m", "pip", "install", "--upgrade", "-r", str(requirements)]
        ).returncode
    if rc != 0:
        log("ERROR: dependency install failed.")
        return 1

    log(f"Done. Invoke directly: {py} <script>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
