"""Strict paired comparison; do not treat a new content domain as a user-population sample."""
import json,pathlib,statistics,hashlib
import sys
root=pathlib.Path(sys.argv[1] if len(sys.argv)>1 else 'evidence/stage22')
files=sorted(root.glob('stdlib_seed*.json'))
if len(files)!=16:raise ValueError(f'incomplete 16-cell study: {len(files)}')
cells={};rows=[]
for p in files:
 d=json.loads(p.read_text());c=d['config'];s=d['summary']
 if d['status']!='measured' or s['offered']!=24 or s['completed']+s['rejected']!=24 or s['errors']!=0:raise ValueError('invalid case: '+str(p))
 key=(c['seed'],c['rate_per_second']);cells.setdefault(key,{})[c['policy']]=d
 if len(cells[key])>4:raise ValueError('duplicate arm')
for key,arms in sorted(cells.items()):
 if set(arms)!={'fifo_static400','sjf_static400','full_adaptive','full_8b_calibrated'}:raise ValueError('missing arms')
 ref=arms['fifo_static400'];reqs=[x['request_id'] for x in ref['requests']]; offsets=ref['config']['arrival_schedule_offsets_s']
 base=ref['summary']['useful_output_tokens_per_second']
 for name,d in arms.items():
  if [x['request_id'] for x in d['requests']]!=reqs or d['config']['arrival_schedule_offsets_s']!=offsets:raise ValueError('not matched')
  s=d['summary'];slo=s['slo_all_offered'];tps=s['useful_output_tokens_per_second'];r=s['rejected'];u=slo['250']['compliance_fraction']-.5*r/24+.2*tps/base
  rows.append(dict(seed=key[0],rate=key[1],policy=name,completed=s['completed'],rejected=r,
       slo250=slo['250']['compliance_fraction'],slo500=slo['500']['compliance_fraction'],slo1000=slo['1000']['compliance_fraction'],tps=tps,utility=u))
result={'status':'pass','cases':len(files),'offers':len(files)*24,'pair_cells':len(cells),'rows':rows,
        'limitations':['Only 2 arrival seeds from one new content domain','PSF CPython docstrings are not sampled production traffic','Single GPU sequential arms; no multi-GPU claims','Model may have memorized stdlib docs during pretraining']}
out=pathlib.Path(sys.argv[2] if len(sys.argv)>2 else 'results/stage22_summary.json');out.write_text(json.dumps(result,indent=2)+'\n')
for rate in (8,16):
 print('RATE',rate)
 for policy in ('fifo_static400','sjf_static400','full_adaptive','full_8b_calibrated'):
  subset=[x for x in rows if x['rate']==rate and x['policy']==policy]
  print(policy,'SLO250',round(statistics.mean(x['slo250'] for x in subset),3),'reject',sum(x['rejected'] for x in subset),'mean_utility',round(statistics.mean(x['utility'] for x in subset),4))
print('STAGE22_ANALYSIS_PASS',len(files),len(files)*24)
