"""~60-second open-loop offers from frozen Stage30 corpus; no duplicated IDs."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from llm_opt_lab.serving import stage30_mixed_bench as bench

RATE=1.6
MIXES=('cold0','hot50')
SEEDS=(3011,3029,3041)

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--seed',type=int,choices=SEEDS,required=True)
 p.add_argument('--mix',choices=MIXES,required=True)
 p.add_argument('--cache',choices=('on','off'),required=True)
 p.add_argument('--output',type=Path,required=True)
 a=p.parse_args(argv)
 original=bench.RATES
 try:
  bench.RATES=(RATE,)
  rc=bench.main(['--seed',str(a.seed),'--mix',a.mix,'--rate',str(RATE),
    '--prefix-cache',a.cache,'--model','qwen3-8b-awq-local','--output',str(a.output)])
 finally:bench.RATES=original
 if rc:return rc
 result=json.loads(a.output.read_text())
 if result['status']!='measured':return 4
 result['kind']='stage33_qwen3_8b_96_unique_offers_sustained_59_375s_cold_hot'
 result['config']['stage33_target_offered_window_s']=(96-1)/RATE
 result['limitations'].append('Target offered duration 59.375s, but demand is 1.6req/s and does not represent overload saturation')
 a.output.write_text(json.dumps(result,indent=2)+'\n')
 return 0
if __name__=='__main__':raise SystemExit(main())
