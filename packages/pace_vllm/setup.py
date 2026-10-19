# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""pace-vllm installer.

Builds pace's C++ runtime via `python setup.py build_clib` against the pace
root, copies the resulting `libpace_cpp.so` into the pace-vllm wheel, and
snapshots `pace/_register_fake.py` as `_fakes_snapshot.py`.
The pace Python package itself is never installed.

The wheel is tagged `py3-none-<plat>` -- there is no CPython extension
module to bind to a specific ABI; the bundled `libpace_cpp.so` is loaded at
runtime via `torch.ops.load_library`.
"""

import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import setup
from setuptools.dist import Distribution
from setuptools.command.build_py import build_py as _build_py

# setuptools >= 70 ships bdist_wheel under setuptools.command; older releases
# only expose it via the standalone `wheel` package. Support both.
try:
    from setuptools.command.bdist_wheel import bdist_wheel as _bdist_wheel
except ImportError:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
_PACE_SETUP = _REPO_ROOT / "setup.py"
_PACE_REGISTER_FAKE = _REPO_ROOT / "pace" / "_register_fake.py"
_PACE_BUILD_PKG_DIR = _REPO_ROOT / "build" / "Release" / "pace"


def _is_monorepo_folder_install() -> bool:
    """True when being installed from the packages/pace_vllm folder."""
    return _PACE_SETUP.exists() and _PACE_REGISTER_FAKE.exists()


def _find_pace_artifact() -> "Path | None":
    """Return libpace_cpp.so produced by the pace CMake build, or None if
    the build hasn't run yet."""
    libpace = _PACE_BUILD_PKG_DIR / "lib" / "libpace_cpp.so"
    return libpace if libpace.exists() else None


def _check_torch_available() -> None:
    """Require Torch for the build; vLLM is checked when the plugin registers."""
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "pace-vllm: torch is not installed; see build_requirements.txt."
        ) from exc


def _build_pace_cpp_only() -> None:
    """Run `python setup.py build_clib` against the pace root.

    Only `build_clib` is needed: pace no longer ships a CPython extension
    module, so there is no `build_ext` step to invoke.
    """
    subprocess.check_call(
        [sys.executable, str(_PACE_SETUP), "build_clib"],
        cwd=str(_REPO_ROOT),
    )


def _place_artifacts(build_pkg_dir: Path) -> None:
    """Copy libpace_cpp.so + fake-op snapshot into `build/lib/pace_vllm/`."""
    libpace = _find_pace_artifact()
    if libpace is None:
        raise RuntimeError(
            "pace-vllm: could not locate libpace_cpp.so at "
            f"{_PACE_BUILD_PKG_DIR}/lib. Did `build_clib` succeed?"
        )

    build_pkg_dir.mkdir(parents=True, exist_ok=True)
    (build_pkg_dir / "lib").mkdir(exist_ok=True)
    shutil.copy2(libpace, build_pkg_dir / "lib" / "libpace_cpp.so")
    shutil.copy2(_PACE_REGISTER_FAKE, build_pkg_dir / "_fakes_snapshot.py")


class BuildPy(_build_py):
    def run(self):
        # Wipe the destination so files removed from source don't ship stale.
        build_pkg_dir = Path(self.build_lib) / "pace_vllm"
        shutil.rmtree(build_pkg_dir, ignore_errors=True)

        # Guard before super().run() so a failed install can't leak
        # partial source files into build_lib/pace_vllm/.
        if not _is_monorepo_folder_install():
            raise RuntimeError(
                "pace-vllm: out-of-monorepo installs are not supported yet. "
                "Install from `packages/pace_vllm` inside amd-pace."
            )
        _check_torch_available()

        super().run()

        _build_pace_cpp_only()
        _place_artifacts(build_pkg_dir)


# Force a platform-tagged wheel even though setuptools sees no ext_modules.
# Without this the wheel would be tagged `any` (pure-Python) and would not
# carry libpace_cpp.so's manylinux/linux platform constraint.
class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


# Override the wheel tag to `py3-none-<plat>`: the wheel ships no CPython
# extension module (libpace_cpp.so links only torch_cpu+c10, no Python C
# API), so it is CPython-version-agnostic. The minimum Python version is
# enforced by `requires-python` in pyproject.toml, not by the wheel tag.
class BdistWheel(_bdist_wheel):
    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self):
        _, _, plat = super().get_tag()
        return ("py3", "none", plat)


setup(
    distclass=BinaryDistribution,
    cmdclass={"build_py": BuildPy, "bdist_wheel": BdistWheel},
)
