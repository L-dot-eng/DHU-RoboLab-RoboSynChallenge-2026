"""Run the official evaluator; fast-exit only after normal completion and worker closure."""
from pathlib import Path
import os
import runpy
import sys


def main():
    import policy.robosyn_team as policy
    models=[]
    original=policy.get_model
    def capture_model(args):
        model=original(args)
        models.append(model)
        return model
    policy.get_model=capture_model
    try:
        runpy.run_path(str(Path(__file__).resolve().parent/'scripts/eval_policy.py'),run_name='__main__')
    finally:
        policy.get_model=original
        for model in models:policy.close_model(model)


def cli():
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    # EmbodiChain normally fast-exits from env.close(). Delay it until metrics
    # have been written; never run this branch after an evaluation/close error.
    os._exit(0)


if __name__=='__main__':cli()
