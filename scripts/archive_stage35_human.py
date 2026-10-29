"""Archive measured OASST1 prompt stats without publishing raw community user text."""
from __future__ import annotations
import argparse,hashlib,json,pathlib,shutil
from analysis.stage35_human_analysis import analyze

def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=pathlib.Path,required=True)
 p.add_argument('--data',type=pathlib.Path,default=pathlib.Path('data'))
 p.add_argument('--out',type=pathlib.Path,default=pathlib.Path('evidence/stage35_real_human'))
 a=p.parse_args()
 result=analyze(a.source,a.data)
 a.out.mkdir(parents=True,exist_ok=True)
 for f in sorted(a.source.glob('seed*.json')):shutil.copy2(f,a.out/f.name)
 shutil.copy2(a.source/'source_sha.txt',a.out/'source_sha.txt')
 (a.out/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
 realservers={f.name:{'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'bytes':f.stat().st_size}
  for f in sorted(a.source.glob('server_*.log'))}
 (a.out/'server_log_hashes_local_only.json').write_text(json.dumps({'source':str(a.source),'server_logs':realservers,
   'notes':'Server logs stay local; prompt source text never committed.'},indent=2)+'\n')
 (a.out/'SHA256SUMS').write_text(''.join(hashlib.sha256(f.read_bytes()).hexdigest()+'  '+f.name+'\n'
  for f in sorted(a.out.glob('*')) if f.is_file() and f.name!='SHA256SUMS'))
 lines=['# Stage35 OASST1 real human-request source vLLM cached-prefix comparison','',
  '**Measured**: 24 real Qwen3-8B-AWQ GPU cases ×96 original community user prompts = **2,304 offered/completed**, three arrival seeds, original-source Apache-2.0 licensed OASST1 root questions in English/Spanish/Chinese, 32 per language. Original user text is retained only locally for privacy; GitHub contains public source SHA, deterministic filtering recipe, anonymous source hashes and complete numeric per-request GPU evidence (no raw user text). Exact paired question IDs, system prefix state, arrival time and prompt SHA verified; vLLM prefix cache ON/OFF under same condition. Server restarts per arm, output capped at **128 tokens**.',
 '', '| Fixed real-user prompt treatment | Offered rate | Mean useful output TPS ON vs OFF | Mean all-offered SLO500 change | Mean P95 TTFT change | Output cap hit-rate (ON) |',
 '|---|---:|---:|---:|---:|---:|']
 for k,value in sorted(result['per_condition'].items()):
  s=k.rsplit('_rate',1);mix=s[0];rate=s[1]
  v=value
  lines.append('| '+mix+' | '+rate+'/s | '+f"{v['relative_tps_gain']['mean']*100:+.2f}%"+' | '
   +f"{v['delta_slo500']['mean']*100:+.2f} pp"+' | '
   +f"{v['delta_p95_ttft_ms']['mean']:+.1f} ms"+' | '
   +f"{v['on_cap_fraction']['mean']*100:.1f}%"+' |')
 lines+=['','**Critical negative evidence:** A very high fraction of original OASST user requests hit the 128-token answer cap. Thus even a high 500ms first-token SLO / tokens-per-second does **not** establish that the assistant supplied a complete human-usable answer. The fixed cap intentionally keeps decode work bounded and permits controlled system comparison; Stage37 has independently frozen output384 conditions and actual vLLM KV-prefix hit token counters to evaluate the tradeoff. Do not claim an end-to-end SLA or answer quality until they are resolved.',
 '', 'Unlike Stage30 PSF docstrings, this is public human-contributed conversational content, but it is filtered and evenly sampled in 3 languages: **not** a representative production workload, and the 0/50/100%-common system instructions were synthetically introduced. Cache ON/OFF is matched within each prompt mixture, but comparing hot100 with cold0 directly changes model instructions and is not a pure caching treatment. Runtime includes a single unrelated warmup per fresh server, not a cold GPU boot. Only three independent arrival seeds per condition; all bootstrap CIs are exploratory.',
 '', 'Raw result files and checksums: `evidence/stage35_real_human/`, analysis source `analysis/stage35_human_analysis.py`, frozen source selection `data/stage35_oasst_manifest.json`. Original prompt text stays in ignored Tang `results/stage35_public_data/` to minimize unnecessary republication of possible personal data. No human preference correctness or claim of natural prefix-hit rate.']
 pathlib.Path('docs/STAGE35_MEASURED_RESULTS.md').write_text('\n'.join(lines)+'\n')
 print('STAGE35_ARCHIVED',result['cases'],'cases',result['offered'],'offers',len(realservers),'server logs privately hashed')
 for k,v in result['per_condition'].items():print(k,'TPS%',round(v['relative_tps_gain']['mean']*100,2),'cap%',round(v['on_cap_fraction']['mean']*100,1))
if __name__=='__main__':main()
