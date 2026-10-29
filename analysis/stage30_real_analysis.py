"""Strict 18-arm, 1728-offer matched cold/hot prefix-cache evaluation."""
from __future__ import annotations
import argparse,hashlib,json,math,random,statistics
from pathlib import Path
SEEDS=(3011,3029,3041)
MIXES={'cold0':0,'hot50':48,'hot100':96}
SLOS=('250','500','1000','2000')

def analyze(root:Path)->dict:
 paths=sorted(root.glob('seed*.json'))
 if len(paths)!=18:raise ValueError(f'Stage30 requires 18 complete arms, found {len(paths)}')
 expected_source=(root/'source_sha.txt').read_text().strip()
 if len(expected_source)!=40:raise ValueError('missing pinned source commit')
 manifest=json.loads(Path('data/stage30_manifest.json').read_text())
 cases={};all_pairs=[]
 for file in paths:
  d=json.loads(file.read_text());c=d['config'];s=d['summary']
  k=(c['mix'],c['seed'],c['prefix_caching'])
  if k in cases:raise ValueError('duplicate mix/seed/cache case')
  if c['seed'] not in SEEDS or c['mix'] not in MIXES or c['prefix_caching'] not in ('on','off'):
   raise ValueError('unapproved treatment')
  if d['status']!='measured' or not(s['offered']==s['completed']==96 and s['rejected']==s['errors']==0):
   raise ValueError('invalid benchmark case '+str(file))
  if c['source_requests_sha256']!=manifest['request_jsonl_sha256']:
   raise ValueError('frozen source data drift')
  if d['environment']['git_commit']!=expected_source:
   raise ValueError('source code SHA drift')
  if c['hot_count']!=MIXES[c['mix']] or c['cold_count']!=96-MIXES[c['mix']]:
   raise ValueError('mix composition mismatch')
  if not d['prompt_usage_valid'] or any(x['status']!='pass' for x in d['mode_length_gate'].values()):
   raise ValueError('invalid output length or prompt usage')
  if not d['server_queue_monitor']['samples'] or not d['kv_cache_monitor']['samples']:
   raise ValueError('real server Prometheus metrics unavailable')
  if len(d['requests'])!=96 or len({r['request_id'] for r in d['requests']})!=96:
   raise ValueError('duplicate request IDs')
  if not math.isfinite(s['useful_output_tokens_per_second']):raise ValueError('bad throughput')
  cases[k]=d
 if len(cases)!=18:raise ValueError('incomplete factorial')
 for mix,hotcount in MIXES.items():
  for seed in SEEDS:
   a=cases[(mix,seed,'off')];b=cases[(mix,seed,'on')]
   ca,cb=a['config'],b['config']
   if ca['scheduled_offsets_s']!=cb['scheduled_offsets_s']:raise ValueError('unequal arrival schedule')
   for ra,rb in zip(a['requests'],b['requests']):
    for field in ('request_id','prompt_sha256','output_mode','prefix_state','scheduled_offset_s','target_prompt_tokens'):
     if ra[field]!=rb[field]:raise ValueError(f'mismatch {mix} {seed} field={field}')
   x,y=a['summary'],b['summary']
   item={'mix':mix,'seed':seed,'off_useful_tps':x['useful_output_tokens_per_second'],
      'on_useful_tps':y['useful_output_tokens_per_second'],
      'relative_tps_gain':y['useful_output_tokens_per_second']/x['useful_output_tokens_per_second']-1,
      'delta_p95_arrival_ttft_ms':y['p95_arrival_ttft_ms']-x['p95_arrival_ttft_ms'],
      'off_p95_ttft_ms':x['p95_arrival_ttft_ms'],'on_p95_ttft_ms':y['p95_arrival_ttft_ms'],
      'off_server_peak_waiting':a['server_queue_monitor']['max_waiting'],
      'on_server_peak_waiting':b['server_queue_monitor']['max_waiting'],
      'delta_slo':{t:y['slo_all_offered'][t]['compliance_fraction']-x['slo_all_offered'][t]['compliance_fraction'] for t in SLOS},
      'prefix_classification':{}}
   for prefix in ('hot','cold'):
    ar=[r for r in a['requests'] if r['prefix_state']==prefix]
    br=[r for r in b['requests'] if r['prefix_state']==prefix]
    if len(ar)!=len(br):raise ValueError('cold/hot subset mismatch')
    if not ar:continue
    item['prefix_classification'][prefix]={'off_count':len(ar),'on_count':len(br),
       'off_p95_ttft':sorted(r['ttft_from_due_ms'] for r in ar)[math.ceil(.95*len(ar))-1],
       'on_p95_ttft':sorted(r['ttft_from_due_ms'] for r in br)[math.ceil(.95*len(br))-1],
       'off_slo500':sum(r['ttft_from_due_ms']<=500 for r in ar)/len(ar),
       'on_slo500':sum(r['ttft_from_due_ms']<=500 for r in br)/len(br)}
   all_pairs.append(item)
 aggregate={};rng=random.Random(300030)
 for mix in MIXES:
  rows=[r for r in all_pairs if r['mix']==mix]
  if len(rows)!=3:raise ValueError('missing seed pairs')
  metrics={'relative_tps_gain':lambda r:r['relative_tps_gain'],
    'delta_p95_arrival_ttft_ms':lambda r:r['delta_p95_arrival_ttft_ms'],
    'delta_slo250':lambda r:r['delta_slo']['250'],
    'delta_slo500':lambda r:r['delta_slo']['500'],
    'delta_slo1000':lambda r:r['delta_slo']['1000'],
    'delta_slo2000':lambda r:r['delta_slo']['2000']}
  agg={'paired_seed_count':len(rows),'source':mix}
  for key,fn in metrics.items():
   values=[fn(r) for r in rows]
   sims=sorted(statistics.mean(values[rng.randrange(3)] for _ in range(3)) for _ in range(5000))
   agg[key]={'mean':statistics.mean(values),'paired_seed_values':values,
      'seed_resampling_bootstrap_95pct_interval':[sims[125],sims[4874]]}
  agg['tps_gain_positive_seeds']=sum(r['relative_tps_gain']>0 for r in rows)
  aggregate[mix]=agg
 return {'status':'pass','kind':'stage30_real_1728_offer_three_seed_cold_hot_prefix_comparison',
   'verified_source_sha':expected_source,'cases':18,'offered':1728,'completed':1728,
   'matched_pairs':all_pairs,'aggregate_by_mix':aggregate,
   'limitations':['18 short but 4× longer 96-request fixed-window runs, no actual hours-long live service',
     'Three random arrivals from fixed synthetic stdlib docs; seed-level bootstrap n=3 is very weak evidence',
     'Server fresh between arms and unrelated model warmup, not GPU cold boot or LLM context ingestion cache cold',
     'No direct prefix-cache hit counts measured, so cache-hit ratio is not established',
     'Model training may include docs, mixed cohort is engineered, not a population of human users']}

def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 result=analyze(a.source)
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n')
 print('STAGE30_ANALYSIS_PASS',result['cases'],result['offered'])
 for mix,stats in result['aggregate_by_mix'].items():
  print(mix,'mean TPS gain',round(stats['relative_tps_gain']['mean'],3),
        'SLO500 change',round(stats['delta_slo500']['mean'],3),
        'P95 delta',round(stats['delta_p95_arrival_ttft_ms']['mean'],1),'positive seeds',stats['tps_gain_positive_seeds'])
if __name__=='__main__':main()
