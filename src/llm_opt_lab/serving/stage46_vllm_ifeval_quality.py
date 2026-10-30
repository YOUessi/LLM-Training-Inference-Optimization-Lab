"""Same IFBench60 source, actual AWQ vLLM base vs AWQ+LoRA responses and official rules.

Frozen Stage42 prompt-only router, no benchmark labels available when choosing.
This tests quantization/serving portability of Stage44's offline NF4 result.
"""
from __future__ import annotations
import argparse,hashlib,json,statistics,time,urllib.request
from pathlib import Path
from types import SimpleNamespace
from llm_opt_lab.training.stage42_risk_router import choose_model
from llm_opt_lab.training.stage44_external_ifbench import frozen,official_scoring,MAX_NEW_TOKENS,MODELS
from llm_opt_lab.provenance import environment,write_json_atomic

MODEL_IDS={'base':'qwen3-8b-awq-local','lora_sft':'qwen3-8b-dolly-r16'}
BASE_URL='http://127.0.0.1:8033'

def post(messages,model,max_tokens=384,seed=460046,timeout=150):
 payload={'model':model,'messages':messages,'max_tokens':max_tokens,
    'temperature':0,'stream':False,'seed':seed,
    'chat_template_kwargs':{'enable_thinking':False}}
 request=urllib.request.Request(BASE_URL+'/v1/chat/completions',
    data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'},method='POST')
 t0=time.perf_counter()
 with urllib.request.urlopen(request,timeout=timeout) as response:
  raw=json.loads(response.read())
 elapsed=time.perf_counter()-t0
 if not raw.get('choices'):raise ValueError('missing actual returned vLLM chat messages')
 message=raw['choices'][0].get('message') or {}
 text=message.get('content')
 usage=raw.get('usage') or {}
 tokens=usage.get('completion_tokens')
 if not isinstance(text,str) or not text.strip() or not isinstance(tokens,int) or not 1<=tokens<=max_tokens:
  raise ValueError('invalid vLLM response content, usage or output cap')
 if raw['choices'][0].get('finish_reason') not in ('stop','length'):
  raise ValueError('unexpected completion end reason '+str(raw['choices'][0].get('finish_reason')))
 return text,tokens,elapsed,raw['choices'][0].get('finish_reason')

def generate(args):
 rows,m=frozen(args.private,args.data)
 with urllib.request.urlopen(BASE_URL+'/v1/models',timeout=12) as resp:
  available={x['id'] for x in json.load(resp).get('data',[])}
 if not set(MODEL_IDS.values()).issubset(available):raise ValueError('AWQ dual server registry incomplete')
 # Warm both endpoints with unrelated query, without sampling any IFBench prompt.
 for model in MODEL_IDS.values():
  post([{'role':'user','content':'What is seven plus four? Answer briefly.'}],model,max_tokens=20)
 starts=time.perf_counter();outputs=[]
 for i,row in enumerate(rows):
  route=choose_model(row['prompt'])
  responses={};tokens={};wall={};reason={}
  for name,model in MODEL_IDS.items():
   text,n,elapsed,why=post([{'role':'user','content':row['prompt']}],model,max_tokens=MAX_NEW_TOKENS)
   responses[name]=text;tokens[name]=n;wall[name]=elapsed;reason[name]=why
  responses['request_router']=responses[route.choice]
  tokens['request_router']=tokens[route.choice]
  wall['request_router']=wall[route.choice]
  reason['request_router']=reason[route.choice]
  outputs.append({'key':str(row['key']),'constraint_families':[id.split(':')[0] for id in row['instruction_id_list']],
      'prompt_sha256':hashlib.sha256(row['prompt'].encode()).hexdigest(),
      'route':route.choice,'risk_score':route.risk_score,'router_features':list(route.features),
      'responses':responses,'output_tokens':tokens,'generation_seconds':wall,'finish_reason':reason,
      'response_sha256':{name:hashlib.sha256(text.encode()).hexdigest() for name,text in responses.items()}})
  if (i+1)%10==0:print('STAGE46_REAL_VLLM_IFBENCH_GENERATED',i+1,'/60',flush=True)
 result={'status':'measured','kind':'stage46_awq_vllm_dual_ifbench_external_60_real_response_text',
  'selected_IFBench_source_sha256':m['selected_sha256'],
  'original_IFBench_source_sha256':m['source_original_sha256'],
  'source_IFBench_commit':m['source_commit'],
  'registered_model_ids':sorted(available),'served_model_ids':MODEL_IDS,
  'runtime':'vLLM 0.19.1 loopback Qwen3-8B AWQ FP16 + NF4-trained rank16 LoRA',
  'generated_count':len(outputs),'model_comparisons':list(MODELS),
  'configs':{'do_sample':False,'thinking_disabled':True,'max_new_tokens':MAX_NEW_TOKENS,
     'route_uses_only_user_prompt':True,'zero_IFBench_labels_passed_to_router':True},
  'raw_outputs':outputs,'wall_seconds':time.perf_counter()-starts,
  'limitations':['AWQ base and FP16 serving are materially different from offline NF4/BF16 base used in Stage44',
    'Raw original IFBench crowd-sourced prompts and generated text remain private in local results only',
    'Max384 output tokens can truncate long format tasks; report actual finish reason and cap frequency',
    'Two model outputs are generated sequentially for auditable quality counterfactual; actual routed API calls one model only as in Stage45',
    'One external-source IFBench60 public subset, no model selection on these outputs']}
 args.out.parent.mkdir(parents=True,exist_ok=True)
 write_json_atomic(args.out,result)
 print('STAGE46_DUAL_AWQ_NF4LORA_REAL_IFBENCH_60X2_COMPLETE',len(outputs),flush=True)

def score(args):
 # Reuse the exact vendored official AllenAI grader via Stage44's tested wrapper.
 tmp=SimpleNamespace(data=args.data,private=args.private,generated=args.generated,out=args.out)
 official_scoring(tmp)
 x=json.loads(args.out.read_text())
 g=json.loads(args.generated.read_text())
 if x['status']!='measured' or len(x['public_anonymized_per_prompt_checks'])!=60:
  raise ValueError('official IFBench scoring incomplete')
 if g['kind']!='stage46_awq_vllm_dual_ifbench_external_60_real_response_text':
  raise ValueError('not an actual AWQ vLLM generation')
 x['kind']='stage46_official_IFBench60_AWQ_vLLM_base_plus_LoRA'
 x['actually_used_vllm_registered_model_ids']=g['registered_model_ids']
 x['server_model_ids']=g['served_model_ids']
 x['runtime_base_quantization']='AWQ FP16, not Stage44 NF4 BF16'
 x['reused_unmodified_official_ifbench_scorer']=True
 x['generation_max_tokens']=MAX_NEW_TOKENS
 x['limits_extra']=['Scores must be compared to Stage44 NF4 scores before AWQ deployment quality equivalence is claimed',
     'Stage45 tests actual routed serving at concurrency; this quality test generates two counterfactual responses sequentially']
 args.out.write_text(json.dumps(x,indent=2)+'\n')
 print('STAGE46_OFFICIAL_IFBENCH_AWQ_VERIFIED',x['aggregated'],'CROSS_BENCH_GATE',x['cross_benchmark_go'],flush=True)

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--action',choices=('generate','score'),required=True)
 ap.add_argument('--private',type=Path,required=True)
 ap.add_argument('--data',type=Path,default=Path('data'))
 ap.add_argument('--generated',type=Path)
 ap.add_argument('--out',type=Path,required=True)
 args=ap.parse_args()
 if args.action=='generate':generate(args)
 else:
  if not args.generated:raise ValueError('--generated must be supplied for score')
  score(args)
if __name__=='__main__':main()
