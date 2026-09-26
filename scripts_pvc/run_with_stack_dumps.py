#!/usr/bin/env python3
"""Run a script in-process with periodic stack dumps of every thread (diagnosing silent phases).

    python scripts_pvc/run_with_stack_dumps.py --every 120 -- train_gotcha_dataset_pvc.py <its arguments>

``faulthandler.dump_traceback_later`` writes all thread stacks to stderr every ``--every`` seconds,
with a timestamp line before each dump, so a phase that prints nothing (TRAIN statistics, the
backprojection start) can be located after the fact. Nothing about the target script changes.
"""
import argparse
import faulthandler
import runpy
import sys
import threading
import time


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--every', type=float, default=120.0, help='seconds between stack dumps')
    p.add_argument('script')
    p.add_argument('args', nargs=argparse.REMAINDER)
    a = p.parse_args()
    args = a.args[1:] if a.args[:1] == ['--'] else a.args

    def stamp():
        while True:
            time.sleep(a.every)
            sys.stderr.write(f'\n=== stack dump at {time.strftime("%H:%M:%S")} (every {a.every:g} s) ===\n')
            sys.stderr.flush()
    threading.Thread(target=stamp, daemon=True).start()
    faulthandler.dump_traceback_later(a.every, repeat=True, file=sys.stderr)
    sys.argv = [a.script] + args
    runpy.run_path(a.script, run_name='__main__')


if __name__ == '__main__':
    main()
