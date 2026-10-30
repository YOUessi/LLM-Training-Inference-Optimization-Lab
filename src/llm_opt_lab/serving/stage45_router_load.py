"""Actual dual-adapter vLLM, counterbalanced 72-request per-arm human-prompt load.

A single engine serves both AWQ base and PEFT LoRA. Router choices derive solely
from the original human message and never the model answer or request rating.
"""
from __future__ import annotations
import argparse,collections,hashlib,json,random,statistics,threading,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from llm_opt_lab.serving.stage35_human_bench import frozen
from llm_opt_lab.serving.stage8_arrivals import schedule,sample_kv_metrics
from llm_opt_lab.serving.bench_client import quantile
from llm_opt_lab.serving.bench_client import _endpoint,_request
from llm_opt_lab.training.stage42_risk_router import choose_model
from llm_opt_lab.provenance import environment,write_json_atomic
SEEDS=(4511,4527)
RATES=(4.,8.)
POLICIES=('base','lora_sft','request_router')
MODELS={'base':'qwen3-8b-awq-local','lora_sft':'qwen3-8b-dolly-r16'}
CAP=192
COUNT=72

def prepare(rows,seed):
 if seed not in SEEDS:raise ValueError('unregistered seed')
 groups={lang:[] for lang in ('en','es','zh')}
 for r in rows:groups[r['lang']].append(r)
 selected=[]
 for lang,items in groups.items():
  ranked=sorted(items,key=lambda r:hashlib.sha256(('stage45-static-oasst-20261030'+r['source_text_sha256']).encode()).hexdigest())
  if len(ranked)!=32:raise ValueError('source stratification drift '+lang)
  selected.extend(ranked[:24])
 random.Random(seed).shuffle(selected)
 if len(selected)!=COUNT or len({r['request_id'] for r in selected})!=COUNT:raise ValueError('not 72 unique human queries')
 return selected

def summarise(records,elapsed):
 """Exactly 72 offered, all-error accounting; no Stage30 fixed96 denominator leak."""
 if len(records)!=COUNT or len({r['request_id'] for r in records})!=COUNT:raise ValueError('request identity/count mismatch')
 ok=[r for r in records if r['status']=='ok']
 ttft=[r['ttft_from_due_ms'] for r in ok]
 return {'offered':COUNT,'completed':len(ok),
   'errors':sum(r['status']=='error' for r in records),
   'rejected':sum(r['status']=='rejected' for r in records),
   'completion_fraction':len(ok)/COUNT,
   'slo_all_offered':{str(s):{'compliance_fraction':sum(t<=s for t in ttft)/COUNT,
      'met_count':sum(t<=s for t in ttft),'offered':COUNT} for s in (250,500,1000,2000)},
   'p95_arrival_ttft_ms':quantile(ttft,.95),
   'p99_arrival_ttft_ms':quantile(ttft,.99),
   'p95_client_executor_wait_ms':quantile([r['client_executor_wait_ms'] for r in records],.95),
   'useful_output_tokens':sum(r['completion_tokens'] for r in ok if isinstance(r.get('completion_tokens'),int)),
   'useful_output_tokens_per_second':sum(r['completion_tokens'] for r in ok if isinstance(r.get('completion_tokens'),int))/elapsed,
   'elapsed_s':elapsed}

def main(argv=None):
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--policy',choices=POLICIES,required=True)
 ap.add_argument('--seed',type=int,choices=SEEDS,required=True)
 ap.add_argument('--rate',type=float,choices=RATES,required=True)
 ap.add_argument('--private',type=Path,required=True)
 ap.add_argument('--input',type=Path,default=Path('data'))
 ap.add_argument('--output',type=Path,required=True)
 a=ap.parse_args(argv)
 rows,source,_=frozen(a.input,a.private)
 selected=prepare(rows,a.seed)
 endpoint=_endpoint('http://127.0.0.1:8033')
 import urllib.request
 with urllib.request.urlopen('http://127.0.0.1:8033/v1/models',timeout=10) as res:
  available={x['id'] for x in json.load(res).get('data',[])}
 if not set(MODELS.values()).issubset(available):raise ValueError('dual registry incomplete')
 # Warm both loaded endpoints on unrelated short prompt to avoid cold load advantage.
 for modelid in MODELS.values():
  warm=_request(-1,endpoint=endpoint,model=modelid,prompt='Answer briefly: what is four plus five?',
    max_tokens=20,seed=a.seed,timeout=70.,disable_thinking=True,raw_prompt=True)
  if warm['status']!='ok':raise ValueError('warmup failed '+modelid)
 offers=schedule(a.seed,a.rate,COUNT,kind='fixed_window')
 kv=[];queues=[];stop=threading.Event()
 monitor=threading.Thread(target=sample_kv_metrics,args=('http://127.0.0.1:8033',stop,kv,queues),daemon=True)
 monitor.start();start=time.perf_counter()+.1
 def run(i,row,due):
  entered=time.perf_counter()
  decision=choose_model(row['source_prompt'])
  chosen=(decision.choice if a.policy=='request_router' else a.policy)
  response=_request(i,endpoint=endpoint,model=MODELS[chosen],prompt=row['source_prompt'],
      max_tokens=CAP,timeout=210.,seed=a.seed,disable_thinking=True,raw_prompt=True)
  response.update({'index':i,'request_id':row['request_id'],'prompt_sha256':row['source_text_sha256'],
    'language':row['lang'],'policy':a.policy,'served_model':chosen,
    'detected_constraint_types':list(decision.features),'risk_score':decision.risk_score,
    'output_mode':'original_user_query','prefix_state':'cold',
    'client_executor_wait_ms':max(0.,entered-due)*1000.,'scheduled_offset_s':due-start})
  response['ttft_from_due_ms']=(max(0.,entered-due)*1000.+response['ttft_ms'] if response.get('ttft_ms') is not None else None)
  return response
 try:
  with ThreadPoolExecutor(max_workers=64) as pool:
   futures=[]
   for i,(offset,row) in enumerate(zip(offers,selected)):
    due=start+offset;time.sleep(max(0.,due-time.perf_counter()))
    futures.append(pool.submit(run,i,row,due))
   records=sorted((f.result() for f in as_completed(futures)),key=lambda r:r['index'])
 finally:
  stop.set();monitor.join(timeout=3.)
 elapsed=time.perf_counter()-start
 summary=summarise(records,elapsed)
 valid=(summary['offered']==summary['completed']==COUNT and summary['errors']==summary['rejected']==0 and
   all(isinstance(x.get('completion_tokens'),int) and 0<x['completion_tokens']<=CAP and x.get('ttft_ms') is not None for x in records)
   and len({x['request_id'] for x in records})==COUNT and len(kv)>0 and len(queues)>0)
 by_model=collections.Counter(r['served_model'] for r in records)
 by_lang={lang:{'offered':sum(r['language']==lang for r in records),
   'completed':sum(r['language']==lang and r['status']=='ok' for r in records),
   'hit_output_cap':sum(r['language']==lang and r.get('completion_tokens')==CAP for r in records)} for lang in ('en','es','zh')}
 result={'status':'measured' if valid else 'invalid_or_blocked',
  'kind':'stage45_real_vllm_AWQ_LoRA_router_multilingual_72_offers',
  'environment':environment(),
  'config':{'seed':a.seed,'rate_per_second':a.rate,'policy':a.policy,'offered_count':COUNT,
    'max_output_tokens':CAP,'client_max_workers':64,'source_sha256':source['source_sha256'],
    'selected96_source_sha256':source['private_selected_jsonl_sha256'],
    'scheduled_offsets_s':offers,'source_selection':'fixed 24 original user queries per language (72 total); order shuffled by seed',
    'fresh_server_for_every_arm':True,'cache':'off','max_num_seqs':16,
    'pre_warm':'two unrelated short requests, one to each registered endpoint'},
  'model_registry':sorted(available),'served_model_counts':dict(by_model),
  'summary':summary,'by_language':by_lang,
  'output_cap_hit_fraction':sum(x['status']=='ok' and x['completion_tokens']>=CAP for x in records)/COUNT,
  'server_queue':{'samples':len(queues),'max_waiting':max((x['waiting'] for x in queues),default=None),
    'max_running':max((x['running'] for x in queues),default=None)},
  'KV_cache':{'samples':len(kv),'max_used_fraction':max((x['kv_usage'] for x in kv),default=None)},
  'all_requests':records,
  'limitations':['Real multilingual OASST1 crowd data, 24 selected queries per language, artificial equal language proportions',
    '128/192 token caps truncate many natural answers; report cap hit fraction and do not claim user-answer quality',
    'One AWQ vLLM engine dual base+LoRA; earlier Stage42 rule quality was measured under BF16 NF4 rather than AWQ',
    '2 arrival seeds and short open-loop offered windows; insufficient for production SLA significance',
    'All modes are separate fresh server runs, so small differences include process setup/measurement noise; warmup both models',
    'User request raw text never archived in GitHub; full original remains in authorized local source file']}
 a.output.parent.mkdir(parents=True,exist_ok=True)
 write_json_atomic(a.output,result)
 print('STAGE45_REAL_DUAL_SERVING',a.policy,a.seed,a.rate,'status',result['status'],
   'tps',round(summary['useful_output_tokens_per_second'],2),'SLO500',round(summary['slo_all_offered']['500']['compliance_fraction'],3),
   'P95',round(summary['p95_arrival_ttft_ms'],1),'cap',round(result['output_cap_hit_fraction'],3),
   'routed_models',dict(by_model),flush=True)
 return 0 if valid else 4
if __name__=='__main__':raise SystemExit(main())
