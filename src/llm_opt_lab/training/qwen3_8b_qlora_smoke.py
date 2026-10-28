"""Actual offline Qwen3-8B 4-bit NF4 QLoRA training smoke; no task quality claim."""
from __future__ import annotations
import argparse,hashlib,json,statistics,time
from pathlib import Path

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps',type=int,default=5)
    parser.add_argument('--accum',type=int,choices=(1,2),default=1)
    parser.add_argument('--out',type=Path,required=True)
    a=parser.parse_args(argv)
    if a.steps<3:parser.error('steps must be >=3')
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig
    from peft import LoraConfig,TaskType,get_peft_model,prepare_model_for_kbit_training
    if not torch.cuda.is_available(): raise RuntimeError('real NVIDIA GPU required')
    torch.manual_seed(18018)
    torch.cuda.manual_seed_all(18018)
    model_dir='/home/you/models/Qwen3-8B'
    qconfig=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16)
    base=AutoModelForCausalLM.from_pretrained(model_dir,local_files_only=True,
        device_map={'':0},dtype=torch.bfloat16,quantization_config=qconfig,
        attn_implementation='sdpa')
    base.config.use_cache=False
    base=prepare_model_for_kbit_training(base,use_gradient_checkpointing=False)
    lora=LoraConfig(r=8,lora_alpha=16,lora_dropout=0.0,bias='none',
        target_modules=['q_proj','v_proj'],task_type=TaskType.CAUSAL_LM)
    model=get_peft_model(base,lora)
    tok=AutoTokenizer.from_pretrained(model_dir,local_files_only=True)
    records=['A transformer computes attention across a sequence of tokens.',
      'A language model can be adapted with trainable low-rank matrices.',
      'Gradient accumulation increases the effective batch size across steps.',
      'An inference benchmark must include rejected requests in the SLO denominator.']
    inputs=[tok(t,return_tensors='pt',truncation=True,max_length=96)['input_ids'].cuda() for t in records]
    holdout=tok('Reasoning models use compute during token generation.',return_tensors='pt')['input_ids'].cuda()
    def evaluate():
        model.eval()
        with torch.no_grad():
            x=model(input_ids=holdout,labels=holdout,use_cache=False).loss.item()
        model.train()
        return float(x)
    opt=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=1e-4)
    ntrain=sum(p.numel() for p in model.parameters() if p.requires_grad)
    before=evaluate()
    times=[];losses=[]
    torch.cuda.reset_peak_memory_stats()
    for step in range(a.steps):
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize();start=time.perf_counter()
        for micro in range(a.accum):
            ids=inputs[(step*a.accum+micro)%len(inputs)]
            loss=model(input_ids=ids,labels=ids,use_cache=False).loss/a.accum
            if not torch.isfinite(loss):raise ValueError('non-finite loss')
            loss.backward()
        opt.step();torch.cuda.synchronize()
        if step>=2:times.append(time.perf_counter()-start);losses.append(float(loss.detach())*a.accum)
    after=evaluate()
    adapter=a.out.parent/(a.out.stem+'_adapter')
    model.save_pretrained(adapter,safe_serialization=True)
    files=sorted(p for p in adapter.rglob('*') if p.is_file())
    checks={str(p.relative_to(adapter)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    out=dict(status='measured',kind='qwen3_8b_real_qlora_nf4_training',base_model='Qwen3-8B',
        quantization='bitsandbytes-nf4-double-quant',lora_rank=8,lora_alpha=16,
        trainable_parameters=ntrain,accum=a.accum,effective_batch_sequences=a.accum,
        warmup_steps=2,measured_steps=len(times),median_step_ms=statistics.median(times)*1000,
        samples_per_second=len(times)*a.accum/sum(times),peak_gpu_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        train_loss_last=losses[-1],heldout_loss_before=before,heldout_loss_after=after,
        adapter_folder=adapter.name,adapter_sha256=checks,
        caveats=['Small fixed diagnostic corpus, not task-quality or generalization proof',
                 'Single 16GiB GPU, no multi-GPU FSDP demonstration'])
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))
if __name__=='__main__': main()
