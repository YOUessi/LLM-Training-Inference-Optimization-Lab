"""Strict paired cache ON/OFF validation of 2304 real OASST1 community prompts."""
from __future__ import annotations
import argparse,hashlib,json,random,statistics,math
from pathlib import Path
SEEDS=(3511,3527,3559);MIXES=('cold0','hot50','hot100')

def analyze(root:Path,public_dir:Path):
 meta=json.loads((public_dir/'stage35_oasst_manifest.json').read_text())
 files=sorted(root.glob('seed*.json'))
 if len(files)!=24:raise ValueError('expected 24 complete real GPU cases; found '+str(len(files)))
 sha=(root/'source_sha.txt').read_text().strip()
 if len(sha)!=40:raise ValueError('missing frozen SHA')
 cases={};rows=[]
 for f in files:
  r=json.loads(f.read_text());c=r['config'];s=r['summary']
  key=(c['seed'],c['mix'],int(c['rate_per_second']),c['cache'])
  if key in cases:raise ValueError('duplicate benchmark cell')
  if r['status']!='measured' or not(s['offered']==s['completed']==96 and s['errors']==s['rejected']==0):
   raise ValueError('incomplete raw cell: '+str(f))
  if c['source_sha256']!=meta['source_sha256'] or c['selected_jsonl_sha256']!=meta['private_selected_jsonl_sha256']:
   raise ValueError('human data source drift')
  if r['environment']['git_commit']!=sha:raise ValueError('source code drift')
  if not r['prompt_usage_valid'] or not r['kv_cache_monitor']['samples'] or not r['server_queue_monitor']['samples']:
   raise ValueError('invalid metric coverage')
  if c['output_cap']!=128 or c['hot_count']!={'cold0':0,'hot50':48,'hot100':96}[c['mix']]:
   raise ValueError('frozen cap or hot treatment changed')
  if len(r['requests'])!=96 or len({x['request_id'] for x in r['requests']})!=96:raise ValueError('duplicate user question IDs')
  if not 0<=r['completion_cap_fraction']<=1:raise ValueError('bad cap fraction')
  cases[key]=r
 for seed in SEEDS:
  for mix in MIXES:
   rates=(8,4) if mix=='hot50' else (8,)
   for rate in rates:
    a=cases[(seed,mix,rate,'off')];b=cases[(seed,mix,rate,'on')]
    if a['config']['scheduled_offsets_s']!=b['config']['scheduled_offsets_s']:raise ValueError('arrival pairing mismatch')
    for x,y in zip(a['requests'],b['requests']):
     for k in ('request_id','prompt_sha256','prefix_state','language','scheduled_offset_s','target_prompt_tokens'):
      if x[k]!=y[k]:raise ValueError('text or schedule mismatch '+k)
    as_,bs=a['summary'],b['summary']
    rows.append({'seed':seed,'mix':mix,'rate':rate,
      'off_tps':as_['useful_output_tokens_per_second'],'on_tps':bs['useful_output_tokens_per_second'],
      'relative_tps_gain':bs['useful_output_tokens_per_second']/as_['useful_output_tokens_per_second']-1,
      'delta_slo500':bs['slo_all_offered']['500']['compliance_fraction']-as_['slo_all_offered']['500']['compliance_fraction'],
      'delta_p95_ttft_ms':bs['p95_arrival_ttft_ms']-as_['p95_arrival_ttft_ms'],
      'off_cap_fraction':a['completion_cap_fraction'],'on_cap_fraction':b['completion_cap_fraction'],
      'off_waiting_peak':a['server_queue_monitor']['max_waiting'],'on_waiting_peak':b['server_queue_monitor']['max_waiting'],
      'per_language_delta_slo500':{lang:b['by_language'][lang]['slo500']-a['by_language'][lang]['slo500'] for lang in ('en','es','zh')}})
 if len(rows)!=12:raise ValueError('not 12 matched cache pairs')
 rng=random.Random(350035)
 aggregated={}
 for mix in MIXES:
  for rate in (8,4):
   xs=[x for x in rows if x['mix']==mix and x['rate']==rate]
   if not xs:continue
   if len(xs)!=3:raise ValueError('not 3 independent seed comparisons')
   metrics={}
   for field in ('relative_tps_gain','delta_slo500','delta_p95_ttft_ms','off_cap_fraction','on_cap_fraction'):
    vals=[x[field] for x in xs];sims=sorted(statistics.mean(vals[rng.randrange(3)] for _ in range(3)) for _ in range(4000))
    metrics[field]={'mean':statistics.mean(vals),'per_seed':vals,
      'paired_seed_bootstrap95':[sims[100],sims[3899]]}
   metrics['positive_throughput_seeds']=sum(x['relative_tps_gain']>0 for x in xs)
   aggregated[mix+'_rate'+str(rate)]=metrics
 return {'status':'pass','kind':'stage35_real_OASST1_2304_human_multilingual_prefix_study',
   'cases':24,'offered':2304,'completed':2304,
   'source_sha256':meta['source_sha256'],'human_prompt_selected_sha256':meta['private_selected_jsonl_sha256'],
   'pinned_source_sha':sha,'paired_rows':rows,'per_condition':aggregated,
   'limitations':['Human-authored public prompts, but language balance manually stratified and app shared system prefix engineered',
    '128-token generated-output cap reached in high fraction of prompts, so not a complete-response quality assessment',
    'Only 3 arrival seeds, short fixed 96-offer windows, seed bootstrap exploratory',
    'Raw human text withheld from repo for privacy; source hash/reconstruction script available',
    'No actual Prefix Cache hit count recorded and no end-to-end production SLA claim']}

def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--data',type=Path,default=Path('data'));p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 x=analyze(a.source,a.data);a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(x,indent=2)+'\n')
 print('STAGE35_VALIDATED',x['cases'],x['offered'])
 for k,v in x['per_condition'].items():
  print(k,'TPS',round(v['relative_tps_gain']['mean']*100,2),'SLO500',round(v['delta_slo500']['mean']*100,2),'CAP_FRACTION',round(v['on_cap_fraction']['mean'],3))
if __name__=='__main__':main()
