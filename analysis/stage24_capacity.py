"""Strict matched per-seed vLLM max-num-seqs capacity analysis."""
import argparse,json,pathlib,statistics
p=argparse.ArgumentParser();p.add_argument('--source',type=pathlib.Path,required=True);p.add_argument('--out',type=pathlib.Path,required=True);a=p.parse_args()
files=sorted(a.source.glob('stdlib_seed*_maxseq*_unlimited.json'))
if len(files)!=8:raise ValueError('expected exactly eight real capacity cases')
pairs={};rows=[]
for f in files:
 d=json.loads(f.read_text());c=d['config'];s=d['summary'];cap=c['server_max_num_seqs'];key=(c['seed'],c['rate_per_second'])
 if cap not in (8,16) or d['status']!='measured' or s['offered']!=24 or s['completed']!=24 or s['rejected']!=0 or s['errors']!=0:raise ValueError('invalid real case: '+str(f))
 pair=pairs.setdefault(key,{})
 if cap in pair:raise ValueError('duplicate cap')
 pair[cap]=d
 rows.append(dict(seed=key[0],rate=key[1],cap=cap,slo250=s['slo_all_offered']['250']['compliance_fraction'],
   slo500=s['slo_all_offered']['500']['compliance_fraction'],slo1000=s['slo_all_offered']['1000']['compliance_fraction'],
   useful_tps=s['useful_output_tokens_per_second'],server_max_waiting=d['server_queue_monitor']['max_waiting'],
   server_max_running=d['server_queue_monitor']['max_running']))
if len(pairs)!=4:raise ValueError('missing paired conditions')
changes=[]
for (seed,rate),arms in sorted(pairs.items()):
 if set(arms)!={8,16}:raise ValueError('missing capacity arm')
 x,y=arms[8],arms[16]
 if x['config']['arrival_schedule_offsets_s']!=y['config']['arrival_schedule_offsets_s'] or [z['request_id'] for z in x['requests']]!=[z['request_id'] for z in y['requests']]:raise ValueError('unpaired request/schedule')
 sx,sy=x['summary'],y['summary']
 changes.append(dict(seed=seed,rate=rate,delta_slo250=sy['slo_all_offered']['250']['compliance_fraction']-sx['slo_all_offered']['250']['compliance_fraction'],
   delta_slo500=sy['slo_all_offered']['500']['compliance_fraction']-sx['slo_all_offered']['500']['compliance_fraction'],
   delta_tps=sy['useful_output_tokens_per_second']-sx['useful_output_tokens_per_second']))
report=dict(status='pass',cases=8,offered=192,paired_cells=4,rows=rows,paired_changes_16_minus_8=changes,
 limitations=['Two arrival seeds and one local CPython domain are not population-level inference',
              'Server capacities are executed in sequential engine sessions; load/warmup drift possible',
              'Only one 16GiB NVIDIA RTX 4090 Laptop; no multi-GPU scalability result'])
a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2)+'\n')
print('STAGE24_STRICT_PASS',len(rows),'cases')
for change in changes:print(change)
for rate in (8,16):
 for cap in (8,16):
  subset=[r for r in rows if r['rate']==rate and r['cap']==cap]
  print('RATE_CAP',rate,cap,'SLO250',round(statistics.mean(x['slo250'] for x in subset),3),
      'TPS',round(statistics.mean(x['useful_tps'] for x in subset),2))
