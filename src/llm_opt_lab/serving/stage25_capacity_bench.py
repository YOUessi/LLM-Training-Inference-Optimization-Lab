"""Stage25 true Qwen3-AWQ shared-vs-distinct-prefix serving workload driver."""
from __future__ import annotations
import argparse,json,os
from pathlib import Path
from llm_opt_lab.serving import stage14_mixed_bench as bench
from llm_opt_lab.serving.stage13_policy import AdmissionGate

DOMAINS={name:(Path(f'data/stage25_{name}_requests.jsonl'),
     Path(f'data/stage25_{name}_manifest.json'),(9401,9409)) for name in ('shared','distinct')}

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--domain',choices=tuple(DOMAINS),required=True)
 p.add_argument('--seed',type=int,choices=(9401,9409),required=True)
 p.add_argument('--rate',type=float,choices=(8.0,16.0),required=True)
 p.add_argument('--output-cap',type=int,choices=(128,256),required=True)
 p.add_argument('--output',type=Path,required=True)
 p.add_argument('--model',default='qwen3-8b-awq-local')
 a=p.parse_args(argv)
 before=(bench.DOMAINS,bench.POLICIES,bench.AblationAdmissionGate,bench.OUTPUT_CAP)
 try:
  bench.DOMAINS=DOMAINS
  bench.POLICIES=('unlimited',)
  bench.AblationAdmissionGate=lambda policy:AdmissionGate(policy)
  bench.OUTPUT_CAP=a.output_cap
  result=bench.main(['--domain',a.domain,'--seed',str(a.seed),
    '--rate',str(a.rate),'--policy','unlimited','--model',a.model,'--output',str(a.output)])
 finally:
  bench.DOMAINS,bench.POLICIES,bench.AblationAdmissionGate,bench.OUTPUT_CAP=before
 if result!=0:return result
 payload=json.loads(a.output.read_text())
 if payload['status']!='measured':return 2
 payload['kind']='stage25_qwen3_awq_real_capacity_prefix_kv_heterogeneous'
 payload['config']['stage25_workload_domain']=a.domain
 payload['config']['stage25_output_cap']=a.output_cap
 payload['config']['stage25_prefix_caching']=os.environ.get('VLLM_PREFIX_CACHING','unknown')
 payload['config']['stage25_gpu_memory_utilization']=float(os.environ.get('GPU_MEMORY_UTILIZATION','.82'))
 payload['limitations']=['Single 16GiB GPU; fixed-window randomized arrivals not an unconditioned Poisson process',
  'PSF docs with synthetic prompt framing and deliberate shared prefix; not live production requests',
  'Stages/arms run sequentially in independent server sessions; per-case significance unknown',
  'First nonempty SSE content is client TTFT, not a server-side CUDA prefill timer']
 a.output.write_text(json.dumps(payload,indent=2)+'\n')
 return 0
if __name__=='__main__':raise SystemExit(main())
