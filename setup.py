# *******************************************************************************
# Modifications Copyright (c) 2024 Advanced Micro Devices, Inc. All rights
# reserved. Notified per clause 4(b) of the license.
# Portions of this file consist of AI-generated content
# *******************************************************************************

import os
import glob
import sysconfig
from pathlib import Path
from setuptools import setup
from distutils.command.install import install
from distutils.command.clean import clean
from setuptools.command.build_py import build_py
from setuptools.command.build_clib import build_clib

try:
    import torch
    from torch.utils import cpp_extension
    from torch.utils.cpp_extension import BuildExtension, CppExtension
except ModuleNotFoundError:
    raise RuntimeError("PyTorch not found, please install PyTorch to continue.")

try:
    from setuptools_scm import get_version

    PACKAGE_VERSION = get_version(root=".", relative_to=__file__)
except (ImportError, LookupError):
    # Fallback if setuptools_scm is not available or not in a git repo
    PACKAGE_VERSION = "0.0.0+unknown"


# Extract a CMake-compatible version (major.minor.patch only)
# CMake doesn't support PEP 440 local version identifiers
def extract_cmake_compatible_version(version_string):
    """Extract CMake-compatible version from PEP 440 version string."""
    import re

    # Match major.minor.patch at the start of the version
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", version_string)
    if match:
        return f"{match.group(1)}.{match.group(2)}.{match.group(3)}"
    return "0.0.0"


PACKAGE_CMAKE_VERSION = extract_cmake_compatible_version(PACKAGE_VERSION)

PYTORCH_VERSION = torch.__version__
PYTORCH_DIR = os.path.dirname(os.path.abspath(torch.__file__))

BUILD_TYPE = "Release"

PACKAGE_NAME = "pace"
CPP_PACKAGE_NAME = PACKAGE_NAME + "_cpp"
PACKAGE_DIR = os.path.abspath(os.path.dirname(__file__))
PACKAGE_CSRC = os.path.join(PACKAGE_DIR, "csrc")
PACKAGE_BUILD_DIR = os.path.join(PACKAGE_DIR, "build")
PACKAGE_BUILD_TYPE_DIR = os.path.join(PACKAGE_BUILD_DIR, BUILD_TYPE)
PACKAGE_INSALL_DIR = os.path.join(PACKAGE_BUILD_TYPE_DIR, PACKAGE_NAME)


class CPPLibBuild(build_clib):
    def run(self):
        print(torch.__config__.show())
        print("*" * 45 + "\nBuilding CPP library\n" + "*" * 45)
        print(f"PACE Version: {PACKAGE_VERSION}")
        print(f"CMake Version: {PACKAGE_CMAKE_VERSION}")
        if not os.path.exists(PACKAGE_BUILD_DIR):
            os.makedirs(PACKAGE_BUILD_DIR)

        cmake_cmd = "cmake"
        cmake_cmd += f" -B {PACKAGE_BUILD_DIR}"
        cmake_cmd += f" -S {PACKAGE_DIR}"
        cmake_cmd += f" -DCMAKE_BUILD_TYPE={BUILD_TYPE}"
        cmake_cmd += f" -DPACKAGE_NAME={CPP_PACKAGE_NAME}"
        cmake_cmd += f" -DPACKAGE_VERSION={PACKAGE_CMAKE_VERSION}"
        cmake_cmd += f" -DCMAKE_PREFIX_PATH={torch.utils.cmake_prefix_path}"
        cmake_cmd += f" -DPYTHON_INCLUDE_DIR={sysconfig.get_paths()['include']}"
        cmake_cmd += f" -DCMAKE_INSTALL_PREFIX={PACKAGE_INSALL_DIR}"
        if os.system(cmake_cmd):
            raise RuntimeError("Build failed, please check the trace.")

        nproc = os.cpu_count()
        make_cmd = f"make -C {PACKAGE_BUILD_DIR} -j {nproc}"
        if os.system(make_cmd):
            raise RuntimeError("Build failed, please check the trace.")

        make_install_cmd = f"make -C {PACKAGE_BUILD_DIR} install"
        if os.system(make_install_cmd):
            raise RuntimeError("Build failed, please check the trace.")


def create_extension():
    print("*" * 45 + "\nBuilding Python Extension package\n" + "*" * 45)
    cpp_files = [os.path.join(PACKAGE_DIR, "csrc/init.cpp")]
    include_dirs = cpp_extension.include_paths() + [
        os.path.join(PACKAGE_CSRC),
    ]
    library_dirs = [
        os.path.join(PACKAGE_INSALL_DIR, "lib"),
        os.path.join(PYTORCH_DIR, "lib"),
    ]
    libraries = [CPP_PACKAGE_NAME]
    compile_args = [
        # "-Wall",
        # "-Wextra",
        # "-Wno-attributes",
        # "-Wno-ignored-attributes",
        # "-Wno-write-strings",
        # "-Wreturn-type",
    ]

    return CppExtension(
        name="{}._C".format(PACKAGE_NAME),
        language="c++",
        sources=cpp_files,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_compile_args=compile_args,
        extra_link_args=["-Wl,-rpath,$ORIGIN/lib"],
    )


cmdclass = {"build_clib": CPPLibBuild}


def get_src_py_and_dst():
    ret = []
    generated_python_files = glob.glob(
        os.path.join(PACKAGE_DIR, PACKAGE_NAME, "**/*.py"), recursive=True
    )
    for src in generated_python_files:
        dst = os.path.join(
            PACKAGE_INSALL_DIR,
            os.path.relpath(src, os.path.join(PACKAGE_DIR, PACKAGE_NAME)),
        )
        dst_path = Path(dst)
        if not dst_path.parent.exists():
            Path(dst_path.parent).mkdir(parents=True, exist_ok=True)
        ret.append((src, dst))
    return ret


class InstallCmd(install, object):
    def finalize_options(self):
        self.build_lib = os.path.relpath(PACKAGE_BUILD_TYPE_DIR)
        return super(InstallCmd, self).finalize_options()


class CleanCmd(clean, object):
    def run(self):
        from distutils.dir_util import remove_tree

        def _remove(path):
            if os.path.exists(path):
                remove_tree(path)

        _remove(os.path.relpath(PACKAGE_BUILD_DIR))
        _remove(os.path.realpath(PACKAGE_NAME + ".egg-info/"))


class PythonPackageBuild(build_py, object):
    def run(self) -> None:
        ret = get_src_py_and_dst()
        for src, dst in ret:
            self.copy_file(src, dst)
        super(PythonPackageBuild, self).finalize_options()


class ExtBuild(BuildExtension):
    def run(self):
        self.run_command("build_clib")

        self.build_lib = os.path.relpath(PACKAGE_BUILD_TYPE_DIR)
        self.build_temp = os.path.relpath(PACKAGE_BUILD_DIR)
        self.library_dirs.append(os.path.join(PACKAGE_BUILD_TYPE_DIR, "lib"))
        super(ExtBuild, self).run()


cmdclass["install"] = InstallCmd
cmdclass["build_py"] = PythonPackageBuild
cmdclass["build_ext"] = ExtBuild
cmdclass["clean"] = CleanCmd


data_files_list = [
    (
        f"{PACKAGE_NAME}/lib",
        [os.path.join(PACKAGE_INSALL_DIR, "lib/libpace_cpp.so")],
    ),
]

setup(
    name=PACKAGE_NAME,
    data_files=data_files_list,
    packages=[PACKAGE_NAME],
    package_data={PACKAGE_NAME: ["*.so", "lib/*.so", "bin/*.dll", "lib/*.lib"]},
    zip_safe=False,
    ext_modules=[create_extension()],
    cmdclass=cmdclass,
)
