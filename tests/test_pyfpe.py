from auditwheel.wheel_abi import analyze_wheel_abi
from auditwheel.policy import POLICY_PRIORITY_LOWEST


def test_analyze_wheel_abi():
    winfo = analyze_wheel_abi('tests/fpewheel-0.0.0-cp35-cp35m-linux_x86_64.whl')
    assert winfo.sym_tag == 'manylinux2_x86_64'  # for external symbols, it could get manylinux2
    assert winfo.pyfpe_tag == 'linux_x86_64'     # but for having the pyfpe reference, it gets just linux
