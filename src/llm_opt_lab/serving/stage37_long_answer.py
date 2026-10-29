"""Rerun exactly original human prompts with 384 vs Stage35's 128 token cap."""
from __future__ import annotations
import argparse
from pathlib import Path
from llm_opt_lab.serving.stage35_human_bench import main as benchmark

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--seed',type=int,choices=(3511,3527,3559),required=True)
 p.add_argument('--mix',choices=('cold0','hot50'),required=True)
 p.add_argument('--cache',choices=('on','off'),required=True)
 p.add_argument('--private',type=Path,required=True)
 p.add_argument('--output',type=Path,required=True)
 a=p.parse_args()
 return benchmark(['--seed',str(a.seed),'--mix',a.mix,'--cache',a.cache,
    '--rate','4','--max-output','384','--private',str(a.private),'--output',str(a.output)])
if __name__=='__main__':raise SystemExit(main())
