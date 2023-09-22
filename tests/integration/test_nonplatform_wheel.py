from __future__ import annotations

import pathlib
import subprocess

import pytest

HERE = pathlib.Path(__file__).parent.resolve()


@pytest.mark.parametrize("mode", ["repair", "addtag", "show"])
def test_non_platform_wheel_repair(mode):
    wheel = HERE / "plumbum-1.6.8-py2.py3-none-any.whl"
    proc = subprocess.run(
        ["auditwheel", mode, str(wheel)],
        stderr=subprocess.PIPE,
        text=True,
    )
    assert proc.returncode == 0
    assert "This does not look like a platform wheel" in proc.stderr
    assert "AttributeError" not in proc.stderr
