from __future__ import annotations

import importlib
import os
import platform
import subprocess
import sys
import zipfile
from argparse import Namespace
from datetime import datetime, timezone
from os.path import isabs
from pathlib import Path
from unittest import mock
from unittest.mock import Mock

import pytest

from auditwheel import lddtree, main_repair
from auditwheel.libc import Libc
from auditwheel.policy import WheelPolicies
from auditwheel.wheel_abi import analyze_wheel_abi

HERE = Path(__file__).parent.resolve()


@pytest.mark.parametrize(
    ("file", "external_libs", "exclude"),
    [
        ("cffi-1.5.0-cp27-none-linux_x86_64.whl", {"libffi.so.5"}, frozenset()),
        ("cffi-1.5.0-cp27-none-linux_x86_64.whl", set(), frozenset(["libffi.so.5"])),
        (
            "cffi-1.5.0-cp27-none-linux_x86_64.whl",
            {"libffi.so.5"},
            frozenset(["libffi.so.noexist", "libnoexist.so.*"]),
        ),
        (
            "cffi-1.5.0-cp27-none-linux_x86_64.whl",
            set(),
            frozenset(["libffi.so.[4,5]"]),
        ),
        (
            "cffi-1.5.0-cp27-none-linux_x86_64.whl",
            {"libffi.so.5"},
            frozenset(["libffi.so.[6,7]"]),
        ),
        (
            "cffi-1.5.0-cp27-none-linux_x86_64.whl",
            set(),
            frozenset([f"{HERE}/*"]),
        ),
        ("cffi-1.5.0-cp27-none-linux_x86_64.whl", set(), frozenset(["libffi.so.*"])),
        ("cffi-1.5.0-cp27-none-linux_x86_64.whl", set(), frozenset(["*"])),
        (
            "python_snappy-0.5.2-pp260-pypy_41-linux_x86_64.whl",
            {"libsnappy.so.1"},
            frozenset(),
        ),
    ],
)
def test_analyze_wheel_abi(file, external_libs, exclude):
    # If exclude libs contain path, LD_LIBRARY_PATH need to be modified to find the libs
    # `lddtree.load_ld_paths` needs to be reloaded for it's `lru_cache`-ed.
    modify_ld_library_path = any(isabs(e) for e in exclude)

    with pytest.MonkeyPatch.context() as cp:
        if modify_ld_library_path:
            cp.setenv("LD_LIBRARY_PATH", f"{HERE}")
            importlib.reload(lddtree)

        wheel_policies = WheelPolicies(libc=Libc.GLIBC, arch="x86_64")
        winfo = analyze_wheel_abi(wheel_policies, str(HERE / file), exclude)
        assert (
            set(winfo.external_refs["manylinux_2_5_x86_64"]["libs"]) == external_libs
        ), f"{HERE}, {exclude}, {os.environ}"

    if modify_ld_library_path:
        importlib.reload(lddtree)


def test_analyze_wheel_abi_pyfpe():
    wheel_policies = WheelPolicies(libc=Libc.GLIBC, arch="x86_64")
    winfo = analyze_wheel_abi(
        wheel_policies,
        str(HERE / "fpewheel-0.0.0-cp35-cp35m-linux_x86_64.whl"),
        frozenset(),
    )
    assert (
        winfo.sym_tag == "manylinux_2_5_x86_64"
    )  # for external symbols, it could get manylinux1
    assert (
        winfo.pyfpe_tag == "linux_x86_64"
    )  # but for having the pyfpe reference, it gets just linux


@pytest.mark.skipif(platform.machine() != "x86_64", reason="only checked on x86_64")
def test_wheel_source_date_epoch(tmp_path, monkeypatch):
    wheel_build_path = tmp_path / "wheel"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "-w",
            wheel_build_path,
            HERE / "sample_extension",
        ],
        check=True,
    )

    wheel_path, *_ = list(wheel_build_path.glob("*.whl"))
    wheel_output_path = tmp_path / "out"
    args = Namespace(
        LIB_SDIR=".libs",
        ONLY_PLAT=False,
        PLAT="manylinux_2_5_x86_64",
        STRIP=False,
        UPDATE_TAGS=True,
        WHEEL_DIR=str(wheel_output_path),
        WHEEL_FILE=[str(wheel_path)],
        EXCLUDE=[],
        cmd="repair",
        func=Mock(),
        prog="auditwheel",
        verbose=1,
    )
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "650203200")
    # patchelf might not be available as we aren't running in a manylinux container
    # here. We don't need need it in this test, so just patch it.
    with mock.patch("auditwheel.patcher._verify_patchelf"):
        main_repair.execute(args, Mock())

    output_wheel, *_ = list(wheel_output_path.glob("*.whl"))
    with zipfile.ZipFile(output_wheel) as wheel_file:
        for file in wheel_file.infolist():
            assert (
                datetime(*file.date_time, tzinfo=timezone.utc).timestamp() == 650203200
            )
