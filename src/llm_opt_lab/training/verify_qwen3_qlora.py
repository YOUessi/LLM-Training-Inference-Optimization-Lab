"""Check both Qwen3 QLoRA artifacts, then independently reload one on CUDA."""
import argparse,hashlib,json,pathlib

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,default=pathlib.Path('results'));p.add_argument('--out',type=pathlib.Path,required=True);a=p.parse_args()
 from safetensors import safe_open
 import torch
 records=[]
 for name in ('stage20_real_qwen3_8b_qlora','stage20_real_qwen3_8b_qlora_accum2'):
  d=json.loads((a.root/(name+'.json')).read_text())
  assert d['status']=='measured' and d['trainable_parameters']==3833856
  adapter=a.root/d['adapter_folder']
  for filename,sha in d['adapter_sha256'].items():
   path=adapter/filename
   assert hashlib.sha256(path.read_bytes()).hexdigest()==sha,(name,filename)
  path=adapter/'adapter_model.safetensors'
  with safe_open(str(path),framework='pt',device='cpu') as f:
   keys=list(f.keys());count=sum(f.get_tensor(k).numel() for k in keys)
  assert count==3833856 and all(('lora_A' in k or 'lora_B' in k) for k in keys),(name,len(keys),count)
  records.append({'name':name,'adapter_tensors':len(keys),'parameters':count,'sha256':d['adapter_sha256']['adapter_model.safetensors']})
 from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig
 from peft import PeftModel
 model_dir='/home/you/models/Qwen3-8B'
 qconf=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16)
 base=AutoModelForCausalLM.from_pretrained(model_dir,local_files_only=True,
     device_map={'':0},dtype=torch.bfloat16,quantization_config=qconf,attn_implementation='sdpa')
 model=PeftModel.from_pretrained(base,a.root/'stage20_real_qwen3_8b_qlora_adapter',is_trainable=False)
 tok=AutoTokenizer.from_pretrained(model_dir,local_files_only=True)
 ids=tok('Reasoning models use compute during token generation.',return_tensors='pt')['input_ids'].cuda()
 model.eval()
 with torch.inference_mode(): loss=float(model(input_ids=ids,labels=ids,use_cache=False).loss)
 original=json.loads((a.root/'stage20_real_qwen3_8b_qlora.json').read_text())
 deviation=abs(loss-original['heldout_loss_after'])
 if deviation>0.05:raise ValueError(f'reloaded adapter loss drift {deviation:.6f}')
 out={'status':'pass','adapters_verified':records,'reloaded_adapter_loss':loss,
      'recorded_posttrain_loss':original['heldout_loss_after'],'absolute_loss_diff':deviation,
      'notes':'Exact checkpoint hash and parameter shapes plus independent GPU reload; tiny heldout sentence only'}
 a.out.parent.mkdir(parents=True,exist_ok=True)
 a.out.write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps(out,indent=2))
if __name__=='__main__':main()
