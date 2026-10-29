"""Real user OASST1-root multilingual prompt stream, cold/half-hot/all-hot app system prompt.

Original user text is never copied into returned JSON or committed to GitHub.
The raw 96 messages live in ignored results only after provenance/PII filters.
"""
from __future__ import annotations
import argparse,collections,hashlib,json,random,statistics,sys,threading,time,urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from llm_opt_lab.provenance import environment,write_json_atomic
from llm_opt_lab.serving.bench_client import _endpoint,_request
from llm_opt_lab.serving.stage30_mixed_bench import summarise
from llm_opt_lab.serving.stage8_arrivals import schedule,sample_kv_metrics

N=96
SEEDS=(3511,3527,3559)
MIXES={'cold0':0,'hot50':48,'hot100':96}
RATES=(4.,8.)

def frozen(public_dir:Path,private_dir:Path):
 manifest=json.loads((public_dir/'stage35_oasst_manifest.json').read_text())
 file=private_dir/'stage35_selected_user_prompts.jsonl'
 raw=file.read_bytes()
 if hashlib.sha256(raw).hexdigest()!=manifest['private_selected_jsonl_sha256']:
  raise ValueError('original human user prompts SHA mismatch')
 rows=[json.loads(s) for s in raw.decode().splitlines()]
 if len(rows)!=N or len({r['request_id'] for r in rows})!=N:raise ValueError('not 96 unique human user messages')
 if collections.Counter(r['lang'] for r in rows)!=collections.Counter({'en':32,'es':32,'zh':32}):
  raise ValueError('language balance drift')
 public={x['request_id']:x for x in manifest['public_request_hashes']}
 for row in rows:
  x=public[row['request_id']]
  if hashlib.sha256(row['source_prompt'].encode()).hexdigest()!=x['content_sha256']:
   raise ValueError('content hash mismatch')
  if row['lang']!=x['lang']:raise ValueError('language label drift')
 system=(private_dir/'stage35_shared_system_prompt.txt').read_text()
 if hashlib.sha256(system.encode()).hexdigest()!=manifest['system_prompt_sha256']:
  raise ValueError('shared system prompt drift')
 return rows,manifest,system

def prepare(rows,seed,mix):
 if seed not in SEEDS or mix not in MIXES:raise ValueError('unregistered arms')
 a=list(rows);random.Random(seed).shuffle(a)
 rng=random.Random(seed+50123)
 hot=set()
 if mix=='hot100':hot=set(range(N))
 elif mix=='hot50':
  for language in ('en','es','zh'):
   indices=[i for i,row in enumerate(a) if row['lang']==language]
   hot.update(rng.sample(indices,16))
 out=[]
 for i,row in enumerate(a):
  r=dict(row);r['prefix_state']='hot' if i in hot else 'cold';out.append(r)
 if len(out)!=96 or sum(x['prefix_state']=='hot' for x in out)!=MIXES[mix]:raise ValueError('bad hot mix')
 return out

def prefix_cache_counters(base_url='http://127.0.0.1:8026'):
 """Read actual vLLM Prefix Cache query/hit token counters; not proxy usage."""
 with urllib.request.urlopen(base_url+'/metrics',timeout=5) as response:
  body=response.read().decode('utf-8')
 keys={'vllm:prefix_cache_queries_total':'queries','vllm:prefix_cache_hits_total':'hits'}
 found={}
 for line in body.splitlines():
  if not line or line.startswith('#'):continue
  field=line.split('{',1)[0].split(' ',1)[0]
  if field not in keys:continue
  value=float(line.rsplit(None,1)[-1])
  if value<0 or not __import__('math').isfinite(value):raise ValueError('invalid cache counter')
  found[keys[field]]=value
 if set(found)!={'queries','hits'}:raise ValueError('missing true vLLM cache hit/query counters')
 return found

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--mix',choices=tuple(MIXES),required=True)
 p.add_argument('--seed',type=int,choices=SEEDS,required=True)
 p.add_argument('--rate',type=float,choices=RATES,default=8.)
 p.add_argument('--cache',choices=('on','off'),required=True)
 p.add_argument('--model',default='qwen3-8b-awq-local')
 p.add_argument('--max-output',type=int,choices=(128,384),default=128)
 p.add_argument('--input',type=Path,default=Path('data'))
 p.add_argument('--private',type=Path,required=True)
 p.add_argument('--output',type=Path,required=True)
 a=p.parse_args(argv)
 try:
  import os
  if os.environ.get('VLLM_PREFIX_CACHING')!=a.cache:raise ValueError('server/client cache setting mismatch')
  original,manifest,system=frozen(a.input,a.private)
  rows=prepare(original,a.seed,a.mix)
  from transformers import AutoTokenizer
  tokenizer=AutoTokenizer.from_pretrained('/home/you/models/Qwen3-8B',local_files_only=True)
  target={}
  for r in rows:
   msg=([{'role':'system','content':system}] if r['prefix_state']=='hot' else [])+[
     {'role':'user','content':r['source_prompt']}]
   value=len(tokenizer.apply_chat_template(msg,tokenize=True,add_generation_prompt=True,enable_thinking=False))
   if not (4<=value<=2048-a.max_output):raise ValueError('out of context '+r['request_id'])
   target[r['request_id']]=value
  endpoint=_endpoint('http://127.0.0.1:8026')
  warm=_request(-1,endpoint=endpoint,model=a.model,prompt='What is five plus three?',
     max_tokens=24,timeout=90.,seed=a.seed,disable_thinking=True,raw_prompt=True)
  if warm['status']!='ok':raise ValueError('pre-run unrelated warmup failure')
  before_prefix=prefix_cache_counters()
  offsets=schedule(a.seed,a.rate,N,kind='fixed_window')
  kv=[];queues=[];stop=threading.Event()
  sampler=threading.Thread(target=sample_kv_metrics,args=('http://127.0.0.1:8026',stop,kv,queues),daemon=True)
  sampler.start();start=time.perf_counter()+.1
  def task(i,row,due):
   entered=time.perf_counter()
   x=_request(i,endpoint=endpoint,model=a.model,prompt=row['source_prompt'],
     max_tokens=a.max_output,timeout=240.,seed=a.seed,disable_thinking=True,raw_prompt=True,
     system_prompt=system if row['prefix_state']=='hot' else None)
   x.update({'index':i,'request_id':row['request_id'],'prompt_sha256':row['source_text_sha256'],
      'language':row['lang'],'output_mode':'human_original','prefix_state':row['prefix_state'],
      'target_prompt_tokens':target[row['request_id']],
      'scheduled_offset_s':due-start,'client_executor_wait_ms':max(0.,entered-due)*1000.})
   x['ttft_from_due_ms']=(max(0.,entered-due)*1000.+x['ttft_ms'] if x.get('ttft_ms') is not None else None)
   return x
  try:
   with ThreadPoolExecutor(max_workers=64) as pool:
    tasks=[]
    for i,(offset,row) in enumerate(zip(offsets,rows)):
     due=start+offset;time.sleep(max(0.,due-time.perf_counter()))
     tasks.append(pool.submit(task,i,row,due))
    recorded=sorted((f.result() for f in as_completed(tasks)),key=lambda x:x['index'])
  finally:
   stop.set();sampler.join(timeout=3.)
  elapsed=time.perf_counter()-start
  after_prefix=prefix_cache_counters()
  queries=max(0.,after_prefix['queries']-before_prefix['queries'])
  hits=max(0.,after_prefix['hits']-before_prefix['hits'])
  if hits>queries+1e-6:raise ValueError('impossible cache counters')
  real_hit_rate=(hits/queries if queries>0 else None)
  summary=summarise(recorded,elapsed)
  prompt_usage=all(r['status']=='ok' and isinstance(r.get('prompt_tokens'),int) and
      abs(r['prompt_tokens']-r['target_prompt_tokens'])<=80 and isinstance(r.get('completion_tokens'),int)
      and r['completion_tokens']>0 for r in recorded)
  valid=summary['completed']==N and summary['errors']==summary['rejected']==0 and prompt_usage and bool(kv) and bool(queues)
  by_lang={}
  for lang in ('en','es','zh'):
   subset=[x for x in recorded if x['language']==lang]
   by_lang[lang]={'offered':len(subset),'completed':sum(x['status']=='ok' for x in subset),
     'mean_completion_tokens':statistics.mean(x['completion_tokens'] for x in subset),
     'at_output_cap_fraction':sum(x['completion_tokens']>=a.max_output for x in subset)/len(subset),
     'slo500':sum(x['ttft_from_due_ms']<=500 for x in subset)/len(subset)}
  payload={'status':'measured' if valid else 'invalid_or_blocked',
    'kind':('stage37_longanswer_real_OASST1_multilingual_output384' if a.max_output==384 else
      'stage35_actual_OASST1_human_prompts_multilingual_prefix_cache_onoff'),
    'environment':environment(),
    'config':{'mix':a.mix,'seed':a.seed,'rate_per_second':a.rate,'cache':a.cache,
      'hot_count':MIXES[a.mix],'source_sha256':manifest['source_sha256'],
      'selected_jsonl_sha256':manifest['private_selected_jsonl_sha256'],
      'system_prompt_sha256':manifest['system_prompt_sha256'],
      'output_cap':a.max_output,'scheduled_offsets_s':offsets,
      'raw_user_prompts_committed':False,'source_license':'OpenAssistant Apache-2.0'},
    'summary':summary,'by_language':by_lang,
    'observed_vllm_prefix_cache':{'queries_tokens':queries,'hit_tokens':hits,'hit_fraction':real_hit_rate,
      'before':before_prefix,'after':after_prefix,'definition':'actual vLLM token counters; null when caching queries unavailable'},
    'completion_cap_fraction':sum(r['completion_tokens']>=a.max_output for r in recorded if r['status']=='ok')/max(1,summary['completed']),
    'prompt_usage_valid':prompt_usage,
    'kv_cache_monitor':{'samples':len(kv),'max_used_fraction':max((r['kv_usage'] for r in kv),default=None)},
    'server_queue_monitor':{'samples':len(queues),'max_waiting':max((r['waiting'] for r in queues),default=None),
      'max_running':max((r['running'] for r in queues),default=None)},
    'requests':recorded,
    'limitations':['Public human user original prompts reviewed/filtered for PII and toxicity, but sample of 32 each en/es/zh not natural traffic proportions',
       'Hot system message is artificially assigned to 50/100% of real user messages, not naturally sampled app sessions',
       'Finite output cap can truncate answers and changes throughput token demand','Open-loop 96 offers at fixed-window rate, not production-hour load',
       'Original raw human prompt content remains only on Tang and is not committed to GitHub']}
  write_json_atomic(a.output,payload)
  print('STAGE35',a.mix,a.seed,'rate',a.rate,'cache',a.cache,'status',payload['status'],
    'completed',summary['completed'],'TPS',round(summary['useful_output_tokens_per_second'],2),
    'SLO500',round(summary['slo_all_offered']['500']['compliance_fraction'],3),
    'P95',round(summary['p95_arrival_ttft_ms'],1),'capfrac',round(payload['completion_cap_fraction'],3),flush=True)
  return 0 if valid else 4
 except (RuntimeError,ValueError,KeyError,OSError,TypeError) as e:
  print('STAGE35_INVALID',type(e).__name__,str(e),file=sys.stderr,flush=True)
  return 2
if __name__=='__main__':raise SystemExit(main())
