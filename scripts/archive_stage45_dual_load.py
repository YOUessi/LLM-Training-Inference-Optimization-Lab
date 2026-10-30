"""Publish all real dual-8B engine metrics without user prompt text or server logs."""
from __future__ import annotations
import argparse,hashlib,json,pathlib,shutil
from analysis.stage45_dual_load_analysis import analyze

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--input',type=pathlib.Path,required=True)
 ap.add_argument('--out',type=pathlib.Path,default=pathlib.Path('evidence/stage45_dual_vllm_load'))
 a=ap.parse_args();r=analyze(a.input)
 if not(r['status']=='pass' and r['actual_cells']==12 and r['actual_completed']==864):
  raise ValueError('stage45 incomplete')
 a.out.mkdir(parents=True,exist_ok=True)
 for f in sorted(a.input.glob('seed*.json')):shutil.copy2(f,a.out/f.name)
 shutil.copy2(a.input/'source_sha.txt',a.out/'source_sha.txt')
 (a.out/'analysis.json').write_text(json.dumps(r,indent=2)+'\n')
 local_logs={f.name:{'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'size':f.stat().st_size} for f in sorted(a.input.glob('server_*.log'))}
 if len(local_logs)!=12:raise ValueError('not all real server logs retained locally')
 (a.out/'private_server_log_hashes.json').write_text(json.dumps({'local_logs':str(a.input),
  'server_logs_SHA256':local_logs,'full_engine_logs_remain_on_tang_not_github':True},indent=2)+'\n')
 (a.out/'SHA256SUMS').write_text(''.join(hashlib.sha256(f.read_bytes()).hexdigest()+'  '+f.name+'\n'
  for f in sorted(a.out.iterdir()) if f.is_file() and f.name!='SHA256SUMS'))
 lines=['# Stage45: actual dual vLLM AWQ+LoRA concurrent routing on public original user queries',
  '', '**Executed on Tang single RTX4090 Laptop 16GiB:** 12 real GPU vLLM 0.19.1 dual-model server arms × **72 original user requests each = 864 offered and completed**. Both Qwen3-8B-AWQ base and rank16 SFT LoRA registered in one OpenAI-compatible vLLM model registry. Two fixed arrival seeds4511/4527 × arrival rate4 or8 offers/s × three strategies: always original base, always loaded LoRA, or Stage42 fixed prompt-only constraint classifier selecting only one model per actual request. Exactly same 72 unique original Apache OASST1 crowd prompts across arms (24/en, 24/es, 24/zh), original human query text retained locally only. Fresh GPU model server per arm, both APIs 20-token warmed, prefix cache OFF to isolate LoRA route overhead, 192 token cap and strict 72 complete+real Prometheus server counters per arm. No fake requests, no dropped rejects, source Git commit+SHA-256 immutable.',
  '', '## Paired across exact user prompts, arrival sequences and 2 seeds',
  '', '| Offered rate | Model choice | Mean actual useful decoded tokens/s | All-offered TTFT≤500ms | All-offered TTFT≤1000ms | P95 arrival-to-first-token | 192-token cap hit rate | Server max waiting |',
  '|---|---|---:|---:|---:|---:|---:|---:|']
 for rate,group in sorted(r['by_rate'].items()):
  for mode in ('base','lora_sft','request_router'):
   v=group[mode]
   getter=lambda key:v[key]['mean']
   lines.append(f"| {rate}/s | {mode} | {getter('useful_tps'):.2f} | {getter('slo500')*100:.2f}% | {getter('slo1000')*100:.2f}% | {getter('p95_ttft_ms'):.1f}ms | {getter('output_cap_hit_fraction')*100:.1f}% | {getter('server_peak_waiting'):.1f} |")
 lines+=['','## Actual routed-decoding performance tradeoff against always-base',
  '', '| Offered rate | Router useful TPS relative to base | Router TTFT≤500ms absolute difference |',
  '|---|---:|---:|']
 for rate,group in sorted(r['by_rate'].items()):
  lines.append(f"| {rate}/s | {group['router_vs_base_relative_TPS']['mean']*100:+.2f}% | {group['router_vs_base_slo500_delta']['mean']*100:+.2f} percentage points |")
 lines+=['','**Interpretation:** These are real 8B AWQ vLLM serving numbers, not the earlier offline NF4/BF16 Stage42 IFEval task quality. The AWQ base uses different quantization and FP16 compute and may exhibit different constraint behavior; a separate actual-server IFBench/IFEval quality audit would be required before proposing deployment. Original public OASST1 prompts were already used in Stage35/37 serving studies, not independently sampled production messages. Only **two** arrival seeds, artificial equal language quotas and 72-request windows prevent statistically established production SLO claims. Max192-token cap truncates many natural user answers (see table); high tokens/s or first-token SLO does not imply good complete responses. One engine serving both models is a functional & short-burst stress test, **not** an operator optimization proof or hour-long soak. All raw user text deliberately omitted from GitHub while hashed source and per-request numeric telemetry are published.',
  '', 'Source: `docs/STAGE45_PROTOCOL.md`, `scripts/run_stage45_real_dual_load.sh`, `analysis/stage45_dual_load_analysis.py`, full actual 12 cell JSONs `evidence/stage45_dual_vllm_load/`. Server logs remain on Tang with recorded SHA.']
 pathlib.Path('docs/STAGE45_MEASURED_RESULTS.md').write_text('\n'.join(lines)+'\n')
 print('STAGE45_12_ARM_864_REQUESTS_ARCHIVED',len(local_logs),'server logs',flush=True)
if __name__=='__main__':main()
