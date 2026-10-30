"""Strict real dual vLLM multicondition comparison, 2 seeds and 4/8 offers/s."""
from __future__ import annotations
import argparse,collections,hashlib,json,math,random,statistics
from pathlib import Path
SEEDS=(4511,4527)
RATES=(4,8)
POLICIES=('base','lora_sft','request_router')


def analyze(root:Path):
 paths=sorted(root.glob('seed*.json'))
 if len(paths)!=12:raise ValueError('Stage45 must include 12 actual vLLM cells; observed '+str(len(paths)))
 sha=(root/'source_sha.txt').read_text().strip()
 if len(sha)!=40:raise ValueError('missing source commit')
 data={}
 for p in paths:
  x=json.loads(p.read_text());c=x['config'];s=x['summary']
  key=(c['seed'],int(c['rate_per_second']),c['policy'])
  if key in data or key[0] not in SEEDS or key[1] not in RATES or key[2] not in POLICIES:
   raise ValueError('duplicate or unapproved server arm '+str(key))
  if x['environment']['git_commit']!=sha:raise ValueError('server source SHA drift')
  if x['status']!='measured' or not(s['offered']==s['completed']==c['offered_count']==72 and s['errors']==s['rejected']==0):
   raise ValueError('not 72 actual successfully generated requests '+str(p))
  if not x['KV_cache']['samples'] or not x['server_queue']['samples']:
   raise ValueError('server Prometheus monitoring missing')
  if c['max_output_tokens']!=192 or c['cache']!='off':raise ValueError('cap/cache differs from preregistered')
  if set(x['model_registry'])!={'qwen3-8b-awq-local','qwen3-8b-dolly-r16'}:
   raise ValueError('model API registry invalid')
  if len(x['all_requests'])!=72 or len({r['request_id'] for r in x['all_requests']})!=72:
   raise ValueError('not exactly 72 unique community user requests')
  if set(x['by_language'])!={'en','es','zh'} or any(x['by_language'][l]['offered']!=24 for l in ('en','es','zh')):
   raise ValueError('language stratification changed')
  if c['policy']=='base' and set(x['served_model_counts'])!={'base'}:raise ValueError('base treatment not exclusively original model')
  if c['policy']=='lora_sft' and set(x['served_model_counts'])!={'lora_sft'}:raise ValueError('LoRA treatment not exclusively LoRA')
  data[key]=x
 paired=[]
 for seed in SEEDS:
  for rate in RATES:
   a=data[(seed,rate,'base')];b=data[(seed,rate,'lora_sft')];r=data[(seed,rate,'request_router')]
   for x in (b,r):
    if x['config']['scheduled_offsets_s']!=a['config']['scheduled_offsets_s']:
     raise ValueError('arrival offsets not paired')
    for before,after in zip(a['all_requests'],x['all_requests']):
     for field in ('request_id','prompt_sha256','language','scheduled_offset_s'):
      if before[field]!=after[field]:raise ValueError('not identical human source '+field)
   def metrics(arm):
    s=arm['summary']
    return {'useful_tps':s['useful_output_tokens_per_second'],
      'slo500':s['slo_all_offered']['500']['compliance_fraction'],
      'slo1000':s['slo_all_offered']['1000']['compliance_fraction'],
      'p95_ttft_ms':s['p95_arrival_ttft_ms'],'output_cap_hit_fraction':arm['output_cap_hit_fraction'],
      'server_peak_waiting':arm['server_queue']['max_waiting'],
      'served_model_counts':arm['served_model_counts']}
   vals={k:metrics(v) for k,v in (('base',a),('lora_sft',b),('request_router',r))}
   paired.append({'seed':seed,'rate':rate,'metrics':vals,
    'relative_router_tps_vs_base':vals['request_router']['useful_tps']/vals['base']['useful_tps']-1,
    'relative_router_tps_vs_lora':vals['request_router']['useful_tps']/vals['lora_sft']['useful_tps']-1,
    'router_slo500_minus_base':vals['request_router']['slo500']-vals['base']['slo500'],
    'router_slo500_minus_lora':vals['request_router']['slo500']-vals['lora_sft']['slo500'],
    'router_p95_ttft_minus_base_ms':vals['request_router']['p95_ttft_ms']-vals['base']['p95_ttft_ms']})
 aggreg={}
 for rate in RATES:
  x=[p for p in paired if p['rate']==rate]
  if len(x)!=2:raise ValueError('need two independent schedule seeds')
  aggregator={}
  for model in POLICIES:
   aggregator[model]={metric:{'mean':statistics.mean(z['metrics'][model][metric] for z in x),
     'per_seed':[z['metrics'][model][metric] for z in x]} for metric in
      ('useful_tps','slo500','slo1000','p95_ttft_ms','output_cap_hit_fraction','server_peak_waiting')}
  aggregator['router_vs_base_relative_TPS']={
    'mean':statistics.mean(z['relative_router_tps_vs_base'] for z in x),
    'per_seed':[z['relative_router_tps_vs_base'] for z in x]}
  aggregator['router_vs_base_slo500_delta']={
    'mean':statistics.mean(z['router_slo500_minus_base'] for z in x),
    'per_seed':[z['router_slo500_minus_base'] for z in x]}
  aggreg[str(rate)]=aggregator
 return {'status':'pass','kind':'stage45_actual_awq_lora_dual_vllm_multilingual_concurrent',
  'source_commit':sha,'actual_cells':12,'actual_offered':864,'actual_completed':864,
  'paired_2seeds_2rates':paired,'by_rate':aggreg,
  'limitations':['Only two independent arrival seeds; DO NOT infer population confidence intervals',
   'Fresh vLLM server per arm and runtime/thermal drift may influence differences',
   'Both AWQ base and LoRA share one engine with prefix cache OFF; original offline NF4 IFEval is not a direct deployment quality score',
   'Original user community prompts may contain personal data; raw text remains local, only hashes/safe numeric metrics committed',
   '192-token cap and short bursts; this is not full natural human answer quality or hour-scale production deployment',
   'One engine warmup on both models before measured offers; cold GPU boot effects excluded']}

def main():
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 r=analyze(a.input);a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(r,indent=2)+'\n')
 print('STAGE45_REAL_DUAL_864_AUDIT_PASS',r['actual_cells'],r['actual_offered'])
 for rate,group in r['by_rate'].items():
  print('RATE',rate,'BASE_TPS',round(group['base']['useful_tps']['mean'],2),
   'LORA_TPS',round(group['lora_sft']['useful_tps']['mean'],2),
   'ROUTER_TPS',round(group['request_router']['useful_tps']['mean'],2),
   'ROUTER_SLO500',round(group['request_router']['slo500']['mean'],3),flush=True)
if __name__=='__main__':main()
