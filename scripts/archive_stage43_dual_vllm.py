"""Verify real loopback vLLM AWQ+LoRA evidence and preserve bounded logs."""
from __future__ import annotations
import argparse,hashlib,json,pathlib,shutil,statistics

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--source',type=pathlib.Path,required=True)
 ap.add_argument('--out',type=pathlib.Path,default=pathlib.Path('evidence/stage43_dual_vllm'))
 args=ap.parse_args();r=json.loads((args.source/'real_dual_model_results.json').read_text())
 if not(r['status']=='measured' and r['PSF_prompts']==24 and r['requests']==len(r['request_results'])==72):raise ValueError('not real 72 API requests')
 if set(r['registered_model_ids'])!=set(r['model_ids'].values()):raise ValueError('actual vLLM registry lacks base/LoRA')
 if len({x['request_id'] for x in r['request_results']})!=24:raise ValueError('duplicate input or incomplete pairs')
 for model in ('base','lora_sft','router'):
  subset=[x for x in r['request_results'] if x['variant']==model]
  if len(subset)!=24 or any(x['status']!='ok' or not isinstance(x['completion_tokens'],int) or x['completion_tokens']<1 for x in subset):
   raise ValueError('API failed/rejected/missing '+model)
  if not r['stats'][model]['completed']==24:raise ValueError('model completion drift')
 ordered={x['index']:[] for x in r['request_results']}
 for x in r['request_results']:ordered[x['index']].append(x)
 if any(len(items)!=3 or len({x['prompt_sha256'] for x in items})!=1 for items in ordered.values()):
  raise ValueError('model comparison not truly same question')
 if 'STAGE43_ENGINE_CLEANUP_EXIT:0' not in (pathlib.Path('/home/you/projects/llmopt_stage43_run.log')).read_text():
  raise ValueError('no verified server cleanup')
 args.out.mkdir(exist_ok=True,parents=True)
 for name in ('real_dual_model_results.json','source_sha.txt','gpu_during_server.csv','client.log','server.log'):
  f=args.source/name
  if f.is_file():shutil.copy2(f,args.out/name)
 (args.out/'SHA256SUMS').write_text(''.join(hashlib.sha256(f.read_bytes()).hexdigest()+'  '+f.name+'\n'
      for f in sorted(args.out.glob('*')) if f.is_file() and f.name!='SHA256SUMS'))
 data=r['stats'];lines=['# Stage43 — actual vLLM 0.19.1 loopback-only Qwen3-8B-AWQ base + rank16 LoRA API',
  '', '**Actual executed** on Tang RTX4090 Laptop 16 GiB, real vLLM single-engine serving `qwen3-8b-awq-local` and `qwen3-8b-dolly-r16` PEFT rank16 Q/V adapter. Base+LoRA model registrations were checked from the actual OpenAI-compatible `/v1/models` endpoint, not inferred from config. **72/72 real HTTP streaming calls** finished across 24 identical public PSF-sourced technical prompts (base, LoRA, frozen Stage42 request-only router, 64 max response tokens), zero rejects, server cleanup returned zero. Raw per-request SSE TTFT and elapsed latency, model IDs, server logs, GPU memory snapshot, SHA256 and source commit in `evidence/stage43_dual_vllm/`.',
  '', '| API model mode | Completed | Mean TTFT | P95 TTFT | Mean output tokens | Mean response latency |',
  '|---|---:|---:|---:|---:|---:|']
 for label in ('base','lora_sft','router'):
  x=data[label]
  lines.append(f"| {label} | {x['completed']} | {x['mean_ttft_ms']:.1f}ms | {x['p95_ttft_ms']:.1f}ms | {x['mean_completion_tokens']:.2f} | {x['mean_latency_ms']:.1f}ms |")
 lines+=['','**Do not generalize:** Model’s LoRA was trained on Qwen3-8B with **NF4/BF16** but this serving engine loads **AWQ/FP16** weights, so offline Stage42 model quality/IFEval results cannot be transplanted to this deployment without a new verification. Sequential 24 requests + 64-token output cap do not establish batch-load capacity, SLO at concurrency, full natural answer quality, statistical serving speedup, safety, or real human traffic. The router calls only one target model in deployment; the three-arm sequential comparison adds audit overhead and is not routed serving throughput. Full production-style concurrent 4/s–8/s verification is a remaining gate.','',
 'Sources: `docs/STAGE43_PROTOCOL.md`, `scripts/run_stage43_vllm_lora_server.sh`, `scripts/run_stage43_dual_vllm.sh`, raw `evidence/stage43_dual_vllm/`.']
 pathlib.Path('docs/STAGE43_MEASURED_RESULTS.md').write_text('\n'.join(lines)+'\n')
 print('STAGE43_ARCHIVED_72_REAL_VLLM',data)
if __name__=='__main__':main()
