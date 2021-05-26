from os.path import isfile, exists, abspath, basename

from auditwheel.patcher import Patchelf
from .policy import (load_policies, get_policy_by_name, get_policy_name,
                     get_priority_by_name, POLICY_PRIORITY_HIGHEST)
from .tools import EnvironmentDefault
import argparse
import logging


logger = logging.getLogger(__name__)


def configure_parser(sub_parsers):
    policies = load_policies()
    policy_names = [p['name'] for p in policies]
    policy_names += [alias for p in policies for alias in p['aliases']]
    epilog = """PLATFORMS:
These are the possible target platform tags, as specified by PEP 600.
Note that old, pre-PEP 600 tags are still usable and are listed as aliases
below.
"""
    for p in policies:
        epilog += f"- {p['name']}"
        if len(p['aliases']) > 0:
            epilog += f" (aliased by {', '.join(p['aliases'])})"
        epilog += '\n'
    highest_policy = get_policy_name(POLICY_PRIORITY_HIGHEST)
    help = "Vendor in external shared library dependencies of a wheel."
    p = sub_parsers.add_parser(
        'repair', help=help, description=help, epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('WHEEL_FILE', help='Path to wheel file.')
    p.add_argument(
        '--plat',
        action=EnvironmentDefault,
        metavar='PLATFORM',
        env='AUDITWHEEL_PLAT',
        dest='PLAT',
        help='Desired target platform. See the available platforms under the '
             f'PLATFORMS section below. (default: "{highest_policy}")',
        choices=policy_names,
        default=highest_policy)
    p.add_argument('-L',
                   '--lib-sdir',
                   dest='LIB_SDIR',
                   help=('Subdirectory in packages to store copied libraries.'
                         ' (default: ".libs")'),
                   default='.libs')
    p.add_argument('-w',
                   '--wheel-dir',
                   dest='WHEEL_DIR',
                   type=abspath,
                   help=('Directory to store delocated wheels (default:'
                         ' "wheelhouse/")'),
                   default='wheelhouse/')
    p.add_argument('--no-update-tags',
                   dest='UPDATE_TAGS',
                   action='store_false',
                   help=('Do not update the wheel filename tags and WHEEL info'
                         ' to match the repaired platform tag.'),
                   default=True)
    p.add_argument('--strip',
                   dest='STRIP',
                   action='store_true',
                   help='Strip symbols in the resulting wheel',
                   default=False)
    p.add_argument('--include',
                   dest='INCLUDE',
                   help='Only include these libraries')
    p.add_argument('--exclude',
                   dest='EXCLUDE',
                   help='Exclude these libraries')
    p.add_argument('--only-plat',
                   dest='ONLY_PLAT',
                   action='store_true',
                   help='Do not check for higher policy compatibility',
                   default=False)
    p.set_defaults(func=execute)


def execute(args, p):
    import os
    from .repair import repair_wheel
    from .wheel_abi import analyze_wheel_abi, NonPlatformWheel

    if not isfile(args.WHEEL_FILE):
        p.error('cannot access %s. No such file' % args.WHEEL_FILE)

    logger.info('Repairing %s', basename(args.WHEEL_FILE))

    if not exists(args.WHEEL_DIR):
        os.makedirs(args.WHEEL_DIR)

    try:
        wheel_abi = analyze_wheel_abi(args.WHEEL_FILE)
    except NonPlatformWheel:
        logger.info('This does not look like a platform wheel')
        return 1

    policy = get_policy_by_name(args.PLAT)
    reqd_tag = policy['priority']

    if reqd_tag > get_priority_by_name(wheel_abi.sym_tag):
        msg = ('cannot repair "%s" to "%s" ABI because of the presence '
               'of too-recent versioned symbols. You\'ll need to compile '
               'the wheel on an older toolchain.' %
               (args.WHEEL_FILE, args.PLAT))
        p.error(msg)

    if reqd_tag > get_priority_by_name(wheel_abi.ucs_tag):
        msg = ('cannot repair "%s" to "%s" ABI because it was compiled '
               'against a UCS2 build of Python. You\'ll need to compile '
               'the wheel against a wide-unicode build of Python.' %
               (args.WHEEL_FILE, args.PLAT))
        p.error(msg)

    abis = [policy['name']] + policy['aliases']
    if not args.ONLY_PLAT:
        if reqd_tag < get_priority_by_name(wheel_abi.overall_tag):
            logger.info(('Wheel is eligible for a higher priority tag. '
                         'You requested %s but I have found this wheel is '
                         'eligible for %s.'),
                        args.PLAT, wheel_abi.overall_tag)
            higher_policy = get_policy_by_name(wheel_abi.overall_tag)
            abis = [higher_policy['name']] + higher_policy['aliases'] + abis

    patcher = Patchelf()
    out_wheel = repair_wheel(args.WHEEL_FILE,
                             abis=abis,
                             lib_sdir=args.LIB_SDIR,
                             out_dir=args.WHEEL_DIR,
                             update_tags=args.UPDATE_TAGS,
                             patcher=patcher,
                             strip=args.STRIP,
                             include=(args.INCLUDE or '').split(','),
                             exclude=(args.EXCLUDE or '').split(','),
                             )

    if out_wheel is not None:
        logger.info('\nFixed-up wheel written to %s', out_wheel)
