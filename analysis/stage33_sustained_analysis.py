"""Fail-closed 59.375s offered-window 12-case cold/hot real Qwen3 evaluation."""
from __future__ import annotations
import argparse,json,math,random,statistics
from pathlib import Path
SEEDS=(3011,3029,3041);MIXES=('cold0','hot50');SLOS=('250','500','1000','2000')

def analyze(root:Path):
 paths=sorted(root.glob('seed*.json'))
 if len(paths)!=12:raise ValueError('expected 12 sustained real GPU cases, found '+str(len(paths)))
 sha=(root/'source_sha.txt').read_text().strip()
 if len(sha)!=40:raise ValueError('missing pinned source commit')
 data={}
 for p in paths:
  x=json.loads(p.read_text());c=x['config'];s=x['summary']
  key=(c['seed'],c['mix'],c['prefix_caching'])
  if key in data:raise ValueError('duplicate condition')
  if c['seed'] not in SEEDS or c['mix'] not in MIXES or c['prefix_caching'] not in ('on','off'):
   raise ValueError('unapproved factor')
  if x['kind']!='stage33_qwen3_8b_96_unique_offers_sustained_59_375s_cold_hot' or x['status']!='measured':
   raise ValueError('invalid result type '+str(p))
  if not (s['offered']==s['completed']==96 and s['rejected']==s['errors']==0):raise ValueError('missing real served requests')
  if c['stage33_target_offered_window_s']!=59.375 or c['rate_per_second']!=1.6:
   raise ValueError('not a true planned 59.375s 1.6/s interval')
  if x['environment']['git_commit']!=sha:raise ValueError('git SHA drift')
  if not x['kv_cache_monitor']['samples'] or not x['server_queue_monitor']['samples']:
   raise ValueError('server monitoring unavailable')
  if not x['prompt_usage_valid'] or any(y['status']!='pass' for y in x['mode_length_gate'].values()):
   raise ValueError('output length/real token gate not met')
  if len(x['requests'])!=96 or len({r['request_id'] for r in x['requests']})!=96:
   raise ValueError('duplicate frozen requests')
  data[key]=x
 paired=[]
 for mix in MIXES:
  for seed in SEEDS:
   a=data[(seed,mix,'off')];b=data[(seed,mix,'on')]
   if a['config']['scheduled_offsets_s']!=b['config']['scheduled_offsets_s']:
    raise ValueError('unmatched arrivals')
   for ra,rb in zip(a['requests'],b['requests']):
    if any(ra[k]!=rb[k] for k in ('request_id','prompt_sha256','scheduled_offset_s','output_mode','prefix_state')):
     raise ValueError('unmatched frozen treatment input')
   sa,sb=a['summary'],b['summary']
   paired.append({'seed':seed,'mix':mix,
     'off_tps':sa['useful_output_tokens_per_second'],'on_tps':sb['useful_output_tokens_per_second'],
     'relative_tps_change':sb['useful_output_tokens_per_second']/sa['useful_output_tokens_per_second']-1,
     'delta_p95_ttft_ms':sb['p95_arrival_ttft_ms']-sa['p95_arrival_ttft_ms'],
     'delta_slo':{t:sb['slo_all_offered'][t]['compliance_fraction']-sa['slo_all_offered'][t]['compliance_fraction'] for t in SLOS},
     'off_server_max_waiting':a['server_queue_monitor']['max_waiting'],
     'on_server_max_waiting':b['server_queue_monitor']['max_waiting']})
 agg={};rng=random.Random(330033)
 for mix in MIXES:
  xs=[r for r in paired if r['mix']==mix];assert len(xs)==3
  values={k:[f(r) for r in xs] for k,f in {
   'relative_tps_change':lambda r:r['relative_tps_change'],
   'delta_p95_ttft_ms':lambda r:r['delta_p95_ttft_ms'],
   'delta_slo500':lambda r:r['delta_slo']['500'],
   'delta_slo1000':lambda r:r['delta_slo']['1000']}.items()}
  entry={'paired_seed_count':3,'positive_tps_seeds':sum(r['relative_tps_change']>0 for r in xs)}
  for metric,arr in values.items():
   draws=sorted(statistics.mean(arr[rng.randrange(3)] for _ in range(3)) for _ in range(4000))
   entry[metric]={'mean':statistics.mean(arr),'per_seed':arr,'seed_resampling_95pct_ci':[draws[100],draws[3899]]}
  agg[mix]=entry
 return {'status':'pass','kind':'stage33_real_1152_8b_sustained_59_375s_cache_verification',
  'pinned_source_sha':sha,'real_cases':12,'offered':1152,'completed':1152,
  'paired':paired,'by_mix':agg,
  'limitations':['59.375-second offered window per arm, not an hour-scale production soak',
   'Fixed 96 synthetic PSF stdlib questions and only 3 arrival seeds; seed bootstraps have low resolution',
   'At 1.6/s GPU may not reach overload; do not compare to 8/s as a pure duration effect',
   'No direct cache-hit counter; cold refers to distinct prefix and fresh server, not GPU cold boot']}

def main():
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 result=analyze(a.input);a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n')
 print('STAGE33_SUSTAINED_ANALYSIS_PASS',result['real_cases'],'cases',result['offered'],'offers')
 for key,row in result['by_mix'].items():
  print(key,'TPS_GAIN',round(row['relative_tps_change']['mean']*100,2),'P95_MS_DELTA',round(row['delta_p95_ttft_ms']['mean'],2),
   'POSITIVE_SEEDS',row['positive_tps_seeds'])
if __name__=='__main__':main()
