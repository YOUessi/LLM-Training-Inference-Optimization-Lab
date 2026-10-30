"""Actually serve Qwen3-8B-AWQ base and selected rank16 LoRA, verify API and routing.

LoRA trained over NF4 BF16 but deployed on AWQ base; model behavior may differ;
this is engineering compatibility and runtime measurement, not quality equivalence.
"""
from __future__ import annotations
import argparse,hashlib,json,statistics,time,urllib.request
from pathlib import Path
from llm_opt_lab.serving.bench_client import _endpoint,_request
from llm_opt_lab.training.stage42_risk_router import choose_model
from llm_opt_lab.provenance import environment,write_json_atomic

MODELS={'base':'qwen3-8b-awq-local','lora_sft':'qwen3-8b-dolly-r16'}

def run():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--out',type=Path,required=True)
 args=parser.parse_args()
 with urllib.request.urlopen('http://127.0.0.1:8033/v1/models',timeout=15) as r:
  model_list=json.load(r)
 available={item['id'] for item in model_list.get('data',[])}
 if not set(MODELS.values()).issubset(available):raise ValueError('vLLM has not actually registered both base and LoRA: '+str(sorted(available)))
 data=[json.loads(s) for s in Path('data/stage30_requests.jsonl').read_text().splitlines()]
 if len(data)!=96:raise ValueError('PSF frozen source must have 96 identities')
 data=data[:24]
 records=[]
 for i,item in enumerate(data):
  prompt=item['prompts']['cold'][:1600]
  if not prompt:raise ValueError('empty real workload')
  route=choose_model(prompt)
  for mode in ('base','lora_sft','router'):
   actual=route.choice if mode=='router' else mode
   done=_request(i,endpoint=_endpoint('http://127.0.0.1:8033'),
      model=MODELS[actual],prompt=prompt,max_tokens=64,timeout=75.,seed=4301,
      response_instruction='',disable_thinking=True,raw_prompt=True)
   record={'index':i,'request_id':item['request_id'],'variant':mode,'underlying_model':actual,
      'selected_by':route.features if mode=='router' else [],
      'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),
      'status':done.get('status'),'completion_tokens':done.get('completion_tokens'),
      'ttft_ms':done.get('ttft_ms'),'latency_ms':done.get('latency_ms'),
      'error':done.get('error')}
   records.append(record)
   if record['status']!='ok' or not isinstance(record['completion_tokens'],int):
    raise ValueError('real vLLM dual adapter API failed '+str(record))
  if (i+1)%6==0:print('STAGE43_DUAL_VLLM_COMPLETED',i+1,'/24',flush=True)
 stats={}
 for mode in ('base','lora_sft','router'):
  rows=[x for x in records if x['variant']==mode]
  stats[mode]={'completed':len(rows),'mean_ttft_ms':statistics.mean(x['ttft_ms'] for x in rows),
   'p95_ttft_ms':sorted(x['ttft_ms'] for x in rows)[int(.95*len(rows))-1],
   'mean_completion_tokens':statistics.mean(x['completion_tokens'] for x in rows),
   'mean_latency_ms':statistics.mean(x['latency_ms'] for x in rows)}
 result={'status':'measured','kind':'stage43_real_vllm_AWQ_and_LoRA_registered_dual_backend_smoke',
  'environment':environment(),'model_ids':MODELS,'registered_model_ids':sorted(available),
  'PSF_prompts':24,'requests':len(records),'request_results':records,'stats':stats,
  'limitations':['24 offline PSF public-source questions, sequential responses to isolate API correctness, not production capacity',
   'AWQ base differs from NF4 BF16 base used in earlier local Stage42 quality tests; this smoke does NOT transfer those quality claims',
   'Only 64 max output tokens per request, not complete user-answer quality',
   'vLLM startup/engine cache memory vs model enabled LoRA must be measured separately for capacity claims',
   'The first 24 PSF questions may contain explicit instructions; router decisions are prompt-only and are not benchmark-label-driven']}
 args.out.parent.mkdir(parents=True,exist_ok=True);write_json_atomic(args.out,result)
 print('STAGE43_DUAL_BASE_LORA_REAL_API_PASS',stats,flush=True)
if __name__=='__main__':run()
