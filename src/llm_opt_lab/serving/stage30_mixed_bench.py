"""96 request/arm, 3 seeded warm/cold prefix mixes, real Qwen3 vLLM open-loop.

Server is restarted for every cache ON/OFF + mix + seed arm. No rejected
requests; client and server wait are separate observational quantities.
"""
from __future__ import annotations
import argparse,hashlib,json,random,statistics,threading,time,sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from llm_opt_lab.provenance import environment,write_json_atomic
from llm_opt_lab.serving.bench_client import _endpoint,_request,quantile
from llm_opt_lab.serving.stage8_arrivals import sample_kv_metrics,schedule,output_length_checks

N=96
MIXES={'cold0':0,'hot50':48,'hot100':96}
SEEDS=(3011,3029,3041)
RATES=(8.,)
MODES=('label','detailed')
SLOS=(250,500,1000,2000)

def frozen_rows(folder):
 path=folder/'stage30_requests.jsonl'; manifest=json.loads((folder/'stage30_manifest.json').read_text())
 body=path.read_bytes()
 if hashlib.sha256(body).hexdigest()!=manifest['request_jsonl_sha256']:raise ValueError('frozen corpus SHA mismatch')
 rows=[json.loads(s) for s in body.decode().splitlines()]
 if len(rows)!=N or len({r['request_id'] for r in rows})!=N or manifest['request_count']!=N:
  raise ValueError('unexpected frozen corpus identities')
 if manifest['planned_mixes']!=MIXES or sorted(manifest['generation_instructions'])!=sorted(MODES):raise ValueError('frozen modes/mixes drift')
 return rows,manifest

def prepare(rows,seed,mix):
 if seed not in SEEDS or mix not in MIXES:raise ValueError('unapproved seeded workload')
 shuffled=list(rows);random.Random(seed).shuffle(shuffled)
 k=MIXES[mix]
 if k not in (0,48,96):raise ValueError('mix count not preapproved')
 slots=[]
 if k==96:slots=list(range(N))
 elif k==48:
  rng=random.Random(seed+303030)
  slots=rng.sample(list(range(0,N,2)),24)+rng.sample(list(range(1,N,2)),24)
 hot=set(slots)
 offered=[]
 for i,row in enumerate(shuffled):
  kind='hot' if i in hot else 'cold';mode=MODES[i%2]
  item={'request_id':row['request_id'],'language':'en','topic':row['topic'],
     'bucket':row['bucket'],'source_module':row['primary_module'],
     'output_mode':mode,'prefix_state':kind,
     'target_prompt_tokens':row['calibrated_prompt_tokens_by_prefix'][kind][mode],
     'prompt_sha256':row['prompt_sha256_by_prefix'][kind],
     'prompt_text':row['prompts'][kind]}
  offered.append(item)
 if len(offered)!=96 or sum(x['prefix_state']=='hot' for x in offered)!=k:raise ValueError('mix count mismatch')
 if {m:sum(x['output_mode']==m for x in offered) for m in MODES}!={'label':48,'detailed':48}:raise ValueError('mode imbalance')
 return offered

def summarise(records,elapsed):
 ok=[x for x in records if x['status']=='ok'];errors=[x for x in records if x['status']=='error']
 offered=len(records);ttft=[r['ttft_from_due_ms'] for r in ok]
 if offered!=N or len({r['request_id'] for r in records})!=N:raise ValueError('offered identity mismatch')
 if elapsed<=0:raise ValueError('invalid elapsed')
 nout=sum(int(x['completion_tokens']) for x in ok if isinstance(x.get('completion_tokens'),int))
 return {'offered':offered,'completed':len(ok),'rejected':sum(r['status']=='rejected' for r in records),
  'errors':len(errors),'completion_fraction':len(ok)/N,
  'slo_all_offered':{str(s):{'compliance_fraction':sum(x<=s for x in ttft)/N,'met_count':sum(x<=s for x in ttft),'offered':N} for s in SLOS},
  'p50_arrival_ttft_ms':quantile(ttft,.5),'p95_arrival_ttft_ms':quantile(ttft,.95),
  'p99_arrival_ttft_ms':quantile(ttft,.99),'max_arrival_ttft_ms':max(ttft) if ttft else None,
  'p95_executor_wait_ms':quantile([r['client_executor_wait_ms'] for r in records],.95),
  'useful_output_tokens':nout,'useful_output_tokens_per_second':nout/elapsed,
  'completed_requests_per_second':len(ok)/elapsed,'elapsed_s':elapsed,
  'by_prefix':{kind:{'offered':sum(r['prefix_state']==kind for r in records),
    'completed':sum(r['prefix_state']==kind and r['status']=='ok' for r in records),
    'slo250':sum(r['prefix_state']==kind and r['status']=='ok' and r['ttft_from_due_ms']<=250 for r in records)/max(1,sum(r['prefix_state']==kind for r in records))} for kind in ('hot','cold')}}

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--mix',choices=tuple(MIXES),required=True)
 p.add_argument('--seed',type=int,choices=SEEDS,required=True)
 p.add_argument('--rate',type=float,choices=RATES,required=True)
 p.add_argument('--prefix-cache',choices=('on','off'),required=True)
 p.add_argument('--model',default='qwen3-8b-awq-local')
 p.add_argument('--base-url',default='http://127.0.0.1:8026')
 p.add_argument('--input',type=Path,default=Path('data'))
 p.add_argument('--output',type=Path,required=True)
 a=p.parse_args(argv)
 try:
  if a.prefix_cache!=__import__('os').environ.get('VLLM_PREFIX_CACHING'):
   raise ValueError('cache setting must match server environment')
  rows,manifest=frozen_rows(a.input);items=prepare(rows,a.seed,a.mix)
  endpoint=_endpoint(a.base_url)
  # Warm model via distinct unrelated prompt, never prewarming hot benchmark prefix.
  warm=_request(-1,endpoint=endpoint,model=a.model,prompt='Check: what is two plus two?',
    max_tokens=24,timeout=90.,seed=a.seed,
    response_instruction='Answer in a short sentence.',disable_thinking=True)
  if warm['status']!='ok':raise RuntimeError('warmup failed')
  offsets=schedule(a.seed,a.rate,N,kind='fixed_window')
  stop=threading.Event();kv=[];queues=[]
  monitor=threading.Thread(target=sample_kv_metrics,args=(a.base_url,stop,kv,queues),daemon=True)
  monitor.start();start=time.perf_counter()+.1
  def one(index,item,due):
   entered=time.perf_counter()
   common={'index':index,'request_id':item['request_id'],'prompt_sha256':item['prompt_sha256'],
    'prefix_state':item['prefix_state'],'output_mode':item['output_mode'],
    'bucket':item['bucket'],'source_module':item['source_module'],
    'target_prompt_tokens':item['target_prompt_tokens'],
    'client_executor_wait_ms':max(0.,entered-due)*1000.,
    'scheduled_offset_s':due-start}
   result=_request(index,endpoint=endpoint,model=a.model,prompt=item['prompt_text'],
    response_instruction=manifest['generation_instructions'][item['output_mode']],
    max_tokens=128,seed=a.seed,timeout=180.,disable_thinking=True)
   result.update(common)
   result['ttft_from_due_ms']=(max(0.,entered-due)*1000.+result['ttft_ms'] if result.get('ttft_ms') is not None else None)
   return result
  try:
   with ThreadPoolExecutor(max_workers=64) as pool:
    futures=[]
    for i,(offset,item) in enumerate(zip(offsets,items)):
     due=start+offset
     time.sleep(max(0.,due-time.perf_counter()))
     futures.append(pool.submit(one,i,item,due))
    rec=sorted((f.result() for f in as_completed(futures)),key=lambda r:r['index'])
  finally:
   stop.set();monitor.join(timeout=3.)
  elapsed=time.perf_counter()-start
  sumry=summarise(rec,elapsed)
  mode_gate={mode:output_length_checks(mode,[x for x in rec if x['output_mode']==mode]) for mode in MODES}
  prompt_usage_gate=all(x['status']!='ok' or (
    isinstance(x.get('prompt_tokens'),int) and abs(x['prompt_tokens']-x['target_prompt_tokens'])<=100 and
    isinstance(x.get('completion_tokens'),int)) for x in rec)
  valid=(sumry['completed']==N and sumry['errors']==sumry['rejected']==0 and
    all(g['status']=='pass' for g in mode_gate.values()) and
    prompt_usage_gate and bool(kv) and bool(queues))
  result={'status':'measured' if valid else 'invalid_or_blocked',
   'kind':'stage30_real_qwen3_8b_awq_mixed_cold_hot_long_window',
   'environment':environment(),
   'config':{'mix':a.mix,'hot_count':MIXES[a.mix],'cold_count':N-MIXES[a.mix],
     'prefix_caching':a.prefix_cache,'seed':a.seed,'rate_per_second':a.rate,
     'offered_count':N,'source_requests_sha256':manifest['request_jsonl_sha256'],
     'mode_counts':{'label':48,'detailed':48},'max_output_tokens':128,
     'max_server_sequences':16,'client_max_workers':64,'scheduled_offsets_s':offsets,
     'warmup':'single unrelated 24-token prompt, not common benchmark prefix',
     'fixed_window':'randomized gaps normalized to the exact 95/8 seconds'},
   'summary':sumry,'mode_length_gate':mode_gate,'prompt_usage_valid':prompt_usage_gate,
   'kv_cache_monitor':{'samples':len(kv),'max_used_fraction':max((x['kv_usage'] for x in kv),default=None)},
   'server_queue_monitor':{'samples':len(queues),
    'max_waiting':max((x['waiting'] for x in queues),default=None),
    'max_running':max((x['running'] for x in queues),default=None)},
   'requests':rec,
   'limitations':['Client first streamed content includes client/server queue, not pure GPU prefill',
     'Only 3 seeded 96-offer short-duration windows; not industrial-length production traffic',
     'Fresh server process for every arm, with only unrelated warmup before measured traffic',
     'Engineered PSF-licensed stdlib prompts; hot prefix is deliberately shared, cold means no shared header',
     'Model pretraining may contain these documents; no human-traffic sampling']}
  write_json_atomic(a.output,result)
  print('STAGE30_REAL',a.mix,a.seed,a.prefix_cache,'status',result['status'],
        'offered',sumry['offered'],'completed',sumry['completed'],'SLO250',round(sumry['slo_all_offered']['250']['compliance_fraction'],3),
        'TPS',round(sumry['useful_output_tokens_per_second'],1),'P95',round(sumry['p95_arrival_ttft_ms'],1),
        'len_gates',[(k,v['status']) for k,v in mode_gate.items()],flush=True)
  return 0 if valid else 4
 except (OSError,ValueError,TypeError,KeyError,RuntimeError) as e:
  print('STAGE30_INVALID',type(e).__name__,str(e),file=sys.stderr,flush=True)
  return 2
if __name__=='__main__':raise SystemExit(main())
