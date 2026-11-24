"""Controlled actual DEV worker connection check, before any free pilot."""
import argparse
from pathlib import Path
import sys
from node_client import ALLOWED, connection,invoke,sha
from reference_qkv import edited

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('fixed_source',type=Path);p.add_argument('--measure',action='store_true');a=p.parse_args();base=a.fixed_source.read_bytes();patch=edited(base)
    with connection() as ssh:
        op='develop_measure' if a.measure else 'develop_quality'
        root,rc=invoke(ssh,op,{ALLOWED[0]:patch},sha(patch),{ALLOWED[0]:base},225000)
        print(root,flush=True);sys.exit(rc)
