"""Strict evaluation of frozen Stage25 real Qwen3 serving factorial; no cherry-picking."""
from __future__ import annotations
import argparse,hashlib,json,math,statistics
from pathlib import Path

def analyze(root:Path)->dict:
 paths=sorted(p for p in root.glob('*.json') if p.name.startswith(('shared_s','distinct_s')))
 if len(paths)!=22:raise ValueError(f'incomplete 22-cell factorial: {len(paths)}')
 source=(root/'source_sha.txt').read_text().strip()
 if len(source)!=40:raise ValueError('missing pinned source sha')
 records={};rows=[]
 for f in paths:
  d=json.loads(f.read_text());c=d['config'];s=d['summary']
  if not (d['status']=='measured' and s['offered']==s['completed']==24 and s['rejected']==s['errors']==0):
   raise ValueError('invalid measured case: '+str(f))
  if d['environment']['git_commit']!=source:raise ValueError('source SHA drift in '+str(f))
  cohort=c['stage25_workload_domain'];seed=c['seed'];rate=int(c['rate_per_second']);cap=c['server_max_num_seqs'];cache=c['stage25_prefix_caching'];mem=round(c['stage25_gpu_memory_utilization'],2);output=c['stage25_output_cap']
  key=(cohort,seed,rate,cap,cache,mem,output)
  if key in records:raise ValueError('duplicate case')
  records[key]=d
  t=[float(r['ttft_from_due_ms']) for r in d['requests'] if r['status']=='ok']
  slo=s['slo_all_offered']
  rows.append({'cohort':cohort,'seed':seed,'rate':rate,'cap':cap,'cache':cache,'mem':mem,'output_cap':output,
    'offered':24,'completed':s['completed'],'rejected':s['rejected'],'slo250':slo['250']['compliance_fraction'],
    'slo500':slo['500']['compliance_fraction'],'slo1000':slo['1000']['compliance_fraction'],
    'p95_ttft_from_due_ms':sorted(t)[math.ceil(len(t)*.95)-1],
    'useful_tps':s['useful_output_tokens_per_second'],
    'server_peak_waiting':d['server_queue_monitor']['max_waiting'],
    'kv_max_used_fraction':d['kv_cache_monitor']['max_used_fraction'],
    'mean_actual_completion_tokens':statistics.mean(x['completion_tokens'] for x in d['requests'])})
 def compare(oldkey,newkey,name):
  if oldkey not in records or newkey not in records:raise ValueError('unpaired factor '+name+str((oldkey,newkey)))
  x,y=records[oldkey],records[newkey]
  xc,yc=x['config'],y['config']
  if xc['frozen_requests_sha256']!=yc['frozen_requests_sha256'] or xc['arrival_schedule_offsets_s']!=yc['arrival_schedule_offsets_s']:
   raise ValueError('frozen source/arrival mismatch '+name)
  if [z['request_id'] for z in x['requests']]!=[z['request_id'] for z in y['requests']]:raise ValueError('offer identity mismatch '+name)
  sx,sy=x['summary'],y['summary'];a=sx['slo_all_offered'];b=sy['slo_all_offered']
  return {'name':name,'condition':{'cohort':newkey[0],'seed':newkey[1],'rate':newkey[2],'cap':newkey[3],
                      'cache':newkey[4],'mem':newkey[5],'output_cap':newkey[6]},
     'delta_slo250':b['250']['compliance_fraction']-a['250']['compliance_fraction'],
     'delta_slo500':b['500']['compliance_fraction']-a['500']['compliance_fraction'],
     'delta_slo1000':b['1000']['compliance_fraction']-a['1000']['compliance_fraction'],
     'delta_useful_tps':sy['useful_output_tokens_per_second']-sx['useful_output_tokens_per_second'],
     'relative_useful_tps_change':sy['useful_output_tokens_per_second']/sx['useful_output_tokens_per_second']-1,
     'delta_server_peak_waiting':y['server_queue_monitor']['max_waiting']-x['server_queue_monitor']['max_waiting']}
 groups={}
 for seed in (9401,9409):
  for rate in (8,16):
   for cap in (8,16):
    old=('shared',seed,rate,cap,'off',.82,128);new=('shared',seed,rate,cap,'on',.82,128)
    groups.setdefault('prefix_cache_shared',[]).append(compare(old,new,'prefix_cache_shared'))
   for cache in ('off','on'):
    old=('shared',seed,rate,8,cache,.82,128);new=('shared',seed,rate,16,cache,.82,128)
    groups.setdefault('max_seqs_8_to_16',[]).append(compare(old,new,'max_seqs_8_to_16'))
  old=('shared',seed,16,16,'off',.82,128);new=('shared',seed,16,16,'off',.70,128)
  groups.setdefault('kv_budget_mem082_to070',[]).append(compare(old,new,'kv_budget_mem082_to070'))
  new=('shared',seed,16,16,'off',.82,256)
  groups.setdefault('output_cap128_to256',[]).append(compare(old,new,'output_cap128_to256'))
 groups.setdefault('prefix_cache_distinct',[]).append(compare(
   ('distinct',9401,16,16,'off',.82,128),('distinct',9401,16,16,'on',.82,128),'prefix_cache_distinct'))
 aggregate={}
 for name,items in groups.items():
  aggregate[name]={'paired_conditions':len(items),
     'mean_delta_slo250':statistics.mean(z['delta_slo250'] for z in items),
     'mean_delta_slo500':statistics.mean(z['delta_slo500'] for z in items),
     'mean_delta_useful_tps':statistics.mean(z['delta_useful_tps'] for z in items),
     'mean_relative_useful_tps_change':statistics.mean(z['relative_useful_tps_change'] for z in items),
     'positive_tps_pairs':sum(z['delta_useful_tps']>0 for z in items)}
 return {'status':'pass','kind':'stage25_real_8b_server_capacity_prefix_kv_analysis',
     'source_git_sha':source,'cases':22,'offered_requests':528,'completed_requests':528,
     'factor_averages':aggregate,'factor_pair_details':groups,'rows':rows,
     'limitations':['Sequential engine sessions; factors have execution-order drift, no full randomized design',
       '2 arrival seeds and 24 distinct Python-doc questions per cell; not production workloads',
       'Output length differs by instruction/cap; prefix reuse is engineered, not typical of all traffic',
       'No statistical significance from two seeds and small sessions','Single 16GiB GPU, Qwen3-8B-AWQ quantized model']}

def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 obj=analyze(a.source)
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(obj,indent=2)+'\n')
 print('STAGE25_ANALYSIS_PASS',obj['cases'],obj['offered_requests'])
 for k,v in obj['factor_averages'].items():print(k,v)
if __name__=='__main__':main()
