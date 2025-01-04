from __future__ import annotations

import argparse
import logging
from os.path import abspath, basename, exists, isfile

from auditwheel.patcher import Patchelf

from .policy import WheelPolicies
from .tools import EnvironmentDefault

logger = logging.getLogger(__name__)


def configure_parser(sub_parsers):
    wheel_policy = WheelPolicies()
    policies = wheel_policy.policies
    policy_names = [p["name"] for p in policies]
    policy_names += [alias for p in policies for alias in p["aliases"]]
    epilog = """PLATFORMS:
These are the possible target platform tags, as specified by PEP 600.
Note that old, pre-PEP 600 tags are still usable and are listed as aliases
below.
"""
    for p in policies:
        epilog += f"- {p['name']}"
        if len(p["aliases"]) > 0:
            epilog += f" (aliased by {', '.join(p['aliases'])})"
        epilog += "\n"
    highest_policy = wheel_policy.get_policy_name(wheel_policy.priority_highest)
    help = """Vendor in external shared library dependencies of a wheel.
If multiple wheels are specified, an error processing one
wheel will abort processing of subsequent wheels.
"""
    p = sub_parsers.add_parser(
        "repair",
        help=help,
        description=help,
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("WHEEL_FILE", help="Path to wheel file.", nargs="+")
    p.add_argument(
        "--plat",
        action=EnvironmentDefault,
        required=False,
        metavar="PLATFORM",
        env="AUDITWHEEL_PLAT",
        dest="PLAT",
        help="Desired target platform. See the available platforms under the "
        f'PLATFORMS section below. (default on current arch: "{highest_policy}")',
        choices=policy_names,
        default=None,
    )
    p.add_argument(
        "--best-plat",
        action="store_true",
        dest="BEST_PLAT",
        help="Automatically determine the best target platform.",
    )
    p.add_argument(
        "-L",
        "--lib-sdir",
        dest="LIB_SDIR",
        help=(
            "Subdirectory in packages to store copied libraries." ' (default: ".libs")'
        ),
        default=".libs",
    )
    p.add_argument(
        "-w",
        "--wheel-dir",
        dest="WHEEL_DIR",
        type=abspath,
        help=("Directory to store delocated wheels (default:" ' "wheelhouse/")'),
        default="wheelhouse/",
    )
    p.add_argument(
        "--no-update-tags",
        dest="UPDATE_TAGS",
        action="store_false",
        help=(
            "Do not update the wheel filename tags and WHEEL info"
            " to match the repaired platform tag."
        ),
        default=True,
    )
    p.add_argument(
        "--strip",
        dest="STRIP",
        action="store_true",
        help="Strip symbols in the resulting wheel",
        default=False,
    )
    p.add_argument(
        "--exclude",
        dest="EXCLUDE",
        help="Exclude SONAME from grafting into the resulting wheel "
        "Please make sure wheel metadata reflects your dependencies. "
        "See https://github.com/pypa/auditwheel/pull/411#issuecomment-1500826281 "
        "(can be specified multiple times) "
        "(can contain wildcards, for example libfoo.so.*)",
        action="append",
        default=[],
    )
    p.add_argument(
        "--only-plat",
        dest="ONLY_PLAT",
        action="store_true",
        help="Do not check for higher policy compatibility",
        default=False,
    )
    p.set_defaults(func=execute)


def execute(args, p):
    import os

    from .repair import repair_wheel
    from .wheel_abi import NonPlatformWheel, analyze_wheel_abi

    exclude = frozenset(args.EXCLUDE)
    wheel_policy = WheelPolicies()

    for wheel_file in args.WHEEL_FILE:
        if not isfile(wheel_file):
            p.error("cannot access %s. No such file" % wheel_file)
        wheel_policy.set_platform_from_wheel(wheel_file)

        logger.info("Repairing %s", basename(wheel_file))

        try:
            wheel_abi = analyze_wheel_abi(wheel_policy, wheel_file, exclude)
        except NonPlatformWheel:
            logger.info(NonPlatformWheel.LOG_MESSAGE)
            return 1

        if args.BEST_PLAT:
            if args.PLAT:
                p.error("Cannot specify both --best-plat and --plat")
            args.PLAT = wheel_abi.overall_tag

        if not exists(args.WHEEL_DIR):
            os.makedirs(args.WHEEL_DIR)

        highest_policy = wheel_policy.get_policy_name(wheel_policy.priority_highest)
        if args.PLAT is None:
            args.PLAT = highest_policy
        policy = wheel_policy.get_policy_by_name(args.PLAT)
        reqd_tag = policy["priority"]

        if reqd_tag > wheel_policy.get_priority_by_name(wheel_abi.sym_tag):
            msg = (
                'cannot repair "%s" to "%s" ABI because of the presence '
                "of too-recent versioned symbols. You'll need to compile "
                "the wheel on an older toolchain or pick a newer platform."
                % (wheel_file, args.PLAT)
            )
            p.error(msg)

        if reqd_tag > wheel_policy.get_priority_by_name(wheel_abi.ucs_tag):
            msg = (
                'cannot repair "%s" to "%s" ABI because it was compiled '
                "against a UCS2 build of Python. You'll need to compile "
                "the wheel against a wide-unicode build of Python."
                % (wheel_file, args.PLAT)
            )
            p.error(msg)

        if reqd_tag > wheel_policy.get_priority_by_name(wheel_abi.blacklist_tag):
            msg = (
                'cannot repair "%s" to "%s" ABI because it depends on '
                "black-listed symbols." % (wheel_file, args.PLAT)
            )
            p.error(msg)

        abis = [policy["name"]] + policy["aliases"]
        if not args.ONLY_PLAT:
            if reqd_tag < wheel_policy.get_priority_by_name(wheel_abi.overall_tag):
                logger.info(
                    (
                        "Wheel is eligible for a higher priority tag. "
                        "You requested %s but I have found this wheel is "
                        "eligible for %s."
                    ),
                    args.PLAT,
                    wheel_abi.overall_tag,
                )
                higher_policy = wheel_policy.get_policy_by_name(wheel_abi.overall_tag)
                abis = [higher_policy["name"]] + higher_policy["aliases"] + abis

        patcher = Patchelf()
        out_wheel = repair_wheel(
            wheel_policy,
            wheel_file,
            abis=abis,
            lib_sdir=args.LIB_SDIR,
            out_dir=args.WHEEL_DIR,
            update_tags=args.UPDATE_TAGS,
            patcher=patcher,
            exclude=exclude,
            strip=args.STRIP,
        )

        if out_wheel is not None:
            logger.info("\nFixed-up wheel written to %s", out_wheel)

        if args.BEST_PLAT:
            args.PLAT = None
