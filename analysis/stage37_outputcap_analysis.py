"""384-token OASST human reply cap vs 128-token original paired source; audit truncation."""
from __future__ import annotations
import argparse,json,hashlib,random,statistics
from pathlib import Path
SEEDS=(3511,3527,3559)

def analyze(longdir:Path,shortdir:Path):
 files=sorted(longdir.glob('seed*.json'))
 if len(files)!=12:raise ValueError('Stage37 requires 12 actual long output cells')
 source=(longdir/'source_sha.txt').read_text().strip()
 if len(source)!=40:raise ValueError('missing stage37 source SHA')
 cases={};paired=[];cap_comparisons=[]
 for p in files:
  x=json.loads(p.read_text());s=x['summary'];c=x['config'];key=(c['seed'],c['mix'],c['cache'])
  if key in cases:raise ValueError('duplicate case')
  if not (x['status']=='measured' and s['offered']==s['completed']==96 and s['errors']==s['rejected']==0):raise ValueError('invalid case '+str(p))
  if c['output_cap']!=384 or c['rate_per_second']!=4.:raise ValueError('not preregistered 384 token 4/s')
  cache_stats=x.get('observed_vllm_prefix_cache')
  if not isinstance(cache_stats,dict):raise ValueError('missing direct vLLM token hit counters')
  if c['cache']=='on' and not cache_stats['queries_tokens']>0:raise ValueError('cache-on must have actual token queries')
  if cache_stats['hit_tokens']>cache_stats['queries_tokens']+1e-6:raise ValueError('impossible measured cache hits')
  if c['mix'] not in ('cold0','hot50') or c['seed'] not in SEEDS:raise ValueError('unknown treatment')
  if x['environment']['git_commit']!=source or not x['prompt_usage_valid']:raise ValueError('source/token gate')
  cases[key]=x
 for seed in SEEDS:
  for mix in ('cold0','hot50'):
   a=cases[(seed,mix,'off')];b=cases[(seed,mix,'on')]
   for x,y in zip(a['requests'],b['requests']):
    if any(x[k]!=y[k] for k in ('request_id','prompt_sha256','prefix_state','language','scheduled_offset_s','target_prompt_tokens')):
     raise ValueError('unpaired cache ON/OFF inputs')
   s,t=a['summary'],b['summary']
   paired.append({'seed':seed,'mix':mix,'relative_tps_gain':t['useful_output_tokens_per_second']/s['useful_output_tokens_per_second']-1,
    'delta_slo500':t['slo_all_offered']['500']['compliance_fraction']-s['slo_all_offered']['500']['compliance_fraction'],
    'delta_p95_ttft_ms':t['p95_arrival_ttft_ms']-s['p95_arrival_ttft_ms'],
    'off_hit_cap':a['completion_cap_fraction'],'on_hit_cap':b['completion_cap_fraction'],
    'cache_on_actual_prefix_hit_fraction':b['observed_vllm_prefix_cache']['hit_fraction'],
    'cache_on_hit_tokens':b['observed_vllm_prefix_cache']['hit_tokens'],
    'cache_on_queries_tokens':b['observed_vllm_prefix_cache']['queries_tokens']})
   if mix=='hot50':
    for cache,condition in [('off',a),('on',b)]:
     q=shortdir/f'seed{seed}_hot50_rate4_{cache}.json'
     old=json.loads(q.read_text())
     if old['status']!='measured' or old['config']['output_cap']!=128:raise ValueError('short baseline invalid')
     if old['config']['scheduled_offsets_s']!=condition['config']['scheduled_offsets_s']:raise ValueError('unmatched offer schedule')
     for x,y in zip(old['requests'],condition['requests']):
      if x['request_id']!=y['request_id'] or x['prompt_sha256']!=y['prompt_sha256']:raise ValueError('nonmatched question')
     cap_comparisons.append({'seed':seed,'cache':cache,
       'cap128_fraction':old['completion_cap_fraction'],'cap384_fraction':condition['completion_cap_fraction'],
       'fraction_cap_hits_reduction':old['completion_cap_fraction']-condition['completion_cap_fraction'],
       'mean_real_output_tokens_128':old['summary']['useful_output_tokens']/96,
       'mean_real_output_tokens_384':condition['summary']['useful_output_tokens']/96,
       'delta_p95_ttft_ms':condition['summary']['p95_arrival_ttft_ms']-old['summary']['p95_arrival_ttft_ms']})
 rng=random.Random(370037);aggregate={}
 for mix in ('cold0','hot50'):
  subset=[r for r in paired if r['mix']==mix]
  aggregate[mix]={}
  for field in ('relative_tps_gain','delta_slo500','delta_p95_ttft_ms','off_hit_cap','on_hit_cap','cache_on_actual_prefix_hit_fraction'):
   vals=[r[field] for r in subset];draws=sorted(statistics.mean(vals[rng.randrange(3)] for _ in range(3)) for _ in range(4000))
   aggregate[mix][field]={'mean':statistics.mean(vals),'per_seed':vals,'bootstrap_seed_95pct':[draws[100],draws[3899]]}
 return {'status':'pass','kind':'stage37_longer_real_user_responses_384_cap_cache_and_truncation',
  'cases':12,'offered':1152,'completed':1152,'pinned_source_sha':source,
  'paired_cache_conditions':paired,'summary':aggregate,
  'matched_cap128_vs384_hot50':cap_comparisons,
  'limitations':['384-token output cap still potentially truncates; report actual cap-hit fraction before claiming full natural responses',
    'When comparing output cap128 and384, decode demand changes, so throughput change is NOT a free caching improvement',
    'Source natural user prompts are public crowd data with synthetic shared app system prefix',
    'Only 3 arrival seeds and short 96-offer windows; no model quality or human preference judged here']}

def main():
 p=argparse.ArgumentParser();p.add_argument('--long',type=Path,required=True);p.add_argument('--short',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 x=analyze(a.long,a.short);a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(x,indent=2)+'\n')
 print('STAGE37_VALIDATED',x['cases'],x['offered'])
 for k,v in x['summary'].items():print(k,'TPS_GAIN',round(v['relative_tps_gain']['mean']*100,2),'ON_CAP384',round(v['on_hit_cap']['mean'],3))
 print('CAP_COMPARISONS',x['matched_cap128_vs384_hot50'])
if __name__=='__main__':main()
