"""Archive 384-token OASST answers and direct vLLM prefix hit statistics."""
from __future__ import annotations
import argparse,json,hashlib,pathlib,shutil
from analysis.stage37_outputcap_analysis import analyze

def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=pathlib.Path,required=True)
 p.add_argument('--short',type=pathlib.Path,default=pathlib.Path('evidence/stage35_real_human'))
 p.add_argument('--out',type=pathlib.Path,default=pathlib.Path('evidence/stage37_long_answers'))
 a=p.parse_args()
 result=analyze(a.source,a.short)
 a.out.mkdir(parents=True,exist_ok=True)
 for f in sorted(a.source.glob('seed*.json')):shutil.copy2(f,a.out/f.name)
 shutil.copy2(a.source/'source_sha.txt',a.out/'source_sha.txt')
 (a.out/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
 (a.out/'server_log_hashes_local_only.json').write_text(json.dumps({f.name:{'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'bytes':f.stat().st_size}
    for f in sorted(a.source.glob('server_*.log'))},indent=2)+'\n')
 (a.out/'SHA256SUMS').write_text(''.join(hashlib.sha256(f.read_bytes()).hexdigest()+'  '+f.name+'\n'
    for f in sorted(a.out.glob('*')) if f.is_file() and f.name!='SHA256SUMS'))
 lines=['# Stage37: human OASST1 requests, 384 generated-token cap and actual vLLM prefix-hit counters','',
  '**Actually measured:** 12 cases of identical original, license-cleared filtered OASST1 English/Spanish/Chinese public community user requests, 96 offers/cell = **1,152 completed real Qwen3-8B-AWQ responses**, three frozen arrival seeds, 4 req/s, cold0 (no synthetic common system prompt) or hot50 (common 198-token system prefix in half the 96 requests), cache ON/OFF. 384 output-token budget compared with previous Stage35 128-token benchmark. Fixed Qwen3 model, vLLM fresh server, no-thinking, tokenizer prompt usage, server queue/KV gauges and prompt identity checks. Source text retained locally only; numeric results and file hashes in `evidence/stage37_long_answers/`.',
  '','| Original human prompt mix | Mean ON vs OFF useful TPS | Mean 500ms SLO change | Mean P95 TTFT change | Real vLLM cache hit token fraction ON | Responses reaching 384-token cap (ON) |',
  '|---|---:|---:|---:|---:|---:|']
 for mix,r in result['summary'].items():
  metrics=lambda field:r[field]['mean']
  lines.append('| '+mix+' | '+f"{metrics('relative_tps_gain')*100:+.2f}%"+' | '
   +f"{metrics('delta_slo500')*100:+.2f} pp"+' | '+f"{metrics('delta_p95_ttft_ms'):+.1f} ms"+' | '
   +f"{metrics('cache_on_actual_prefix_hit_fraction')*100:.1f}%"+' | '+f"{metrics('on_hit_cap')*100:.1f}%"+' |')
 old=result['matched_cap128_vs384_hot50']
 if len(old)!=6:raise ValueError('missing 6 honest 128 vs384 matched conditions')
 before=sum(r['cap128_fraction'] for r in old)/len(old)
 after=sum(r['cap384_fraction'] for r in old)/len(old)
 avgshort=sum(r['mean_real_output_tokens_128'] for r in old)/len(old)
 avglong=sum(r['mean_real_output_tokens_384'] for r in old)/len(old)
 lines+=['','**Direct mechanism evidence:** `vllm:prefix_cache_queries_total` and `vllm:prefix_cache_hits_total` are actual Prometheus counters of *queried/cached prefix Tokens*, sampled immediately after an unrelated warmup and again after the fixed 96-offer window; the difference is the actual token hit fraction for caching ON. It is **not** an inferred hit rate from throughput and is `null` for OFF when the engine did not query a cache. The fixed app system prefix is engineered and should not be called an observed naturally-occurring production cache distribution.',
  '',f'**Output truncation audit:** Six matched 4/s hot50 cases compare old cap128 and new cap384, with matching user prompts and arrival schedules but different generated token demand. On average **{before*100:.1f}%** of old responses hit 128 cap versus **{after*100:.1f}%** of new responses hitting 384 cap. Mean realized output length changed from **{avgshort:.1f} to {avglong:.1f}** tokens. If a substantial fraction still hits 384, **do not** claim complete human-usable answers or a production response-quality SLA. Changes in latency/throughput across caps are due partly to decode work changes, not pure caching.',
  '', '**Limits:** Three seeded 96-offer windows at 4/s, public community prompts filtered and artificially language balanced, engineered shared app prefix, no hour-long customer traffic; seed-resampled intervals have low resolution and do not cover model/traffic populations. The original human text is not committed. Prefix hit fraction measures KV prefix token queries, not global per-request prefix cache hit incidence or semantic correctness. Model quality and instruction following are separate Stage36 measurements.',
  '', 'Reproduction: `docs/STAGE37_PROTOCOL.md`, `scripts/run_stage37_long_answers.sh`, `analysis/stage37_outputcap_analysis.py`, `evidence/stage37_long_answers/` and previous `evidence/stage35_real_human/`.']
 pathlib.Path('docs/STAGE37_MEASURED_RESULTS.md').write_text('\n'.join(lines)+'\n')
 print('STAGE37_ARCHIVED',result['cases'],result['offered'],'outputs',before,after,'real_vllm_hit_frac',
  {k:round(v['cache_on_actual_prefix_hit_fraction']['mean'],3) for k,v in result['summary'].items()})
if __name__=='__main__':main()
