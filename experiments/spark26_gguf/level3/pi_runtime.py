"""Shared Pi runtime setup for Level 3 production and fixture entry points."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT), str(HERE.parent / 'v3_dev_rpc')]

import p3_explore as cfg
from verify_common import PI
from evolveengine.v3_level3_guest import runtime_for

PACKAGE_LOCK = ROOT / 'experiments/p2_os_boundary/package-lock.json'


def profile_and_runtime(limits, system, tools=None, *, resources=None):
    profile = dict(cfg.p3.pilot_profile(), maxTokens=limits.per_turn_output)
    return profile, runtime_for(PI, PACKAGE_LOCK.read_bytes(), profile, system, tools, resources=resources)
