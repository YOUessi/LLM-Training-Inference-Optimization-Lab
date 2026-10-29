"""Fail-closed project2 multi-track evidence audit, not an automatic production approval."""
from __future__ import annotations
import argparse,hashlib,json,math
from pathlib import Path
from analysis.stage25_real_analysis import analyze as serving_audit
from analysis.stage26_sft_analysis import analyze as sft_audit

def file_sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def audit(root:Path):
 serving=serving_audit(root/'evidence/stage25_measured')
 if serving['cases']!=22 or serving['completed_requests']!=528:raise ValueError('incomplete serving evidence')
 # Check every hash from frozen Stage25 SHA file, not merely JSON summary.
 serving_root=root/'evidence/stage25_measured'
 for ln in (serving_root/'SHA256SUMS').read_text().splitlines():
  digest,filename=ln.split(None,1)
  if file_sha(serving_root/filename.strip())!=digest:raise ValueError('Stage25 SHA mismatch '+filename)
 sft=sft_audit(root/'evidence/stage26_sft_v2',root/'data')
 if sft['selected']!='B' or sft['test']['selected_ce']>=sft['test']['base_ce']:raise ValueError('SFT holdout failed')
 kernel=json.loads((root/'evidence/stage27_measured/benchmark.json').read_text())
 if kernel['status']!='measured' or len(kernel['cases'])!=8:raise ValueError('kernel evidence missing')
 for c in kernel['cases']:
  if c['numerical_max_abs_error']>c['atol'] or not math.isfinite(c['speedup_vs_compile']):raise ValueError('kernel correctness gate')
  if any(v['median_ms']<=0 for v in c['timings'].values()):raise ValueError('invalid benchmark timing')
 state=json.loads((root/'evidence/stage28_measured/comparison.json').read_text())
 if state['status']!='pass' or len(state['runs'])!=4 or not state['exact_final_adapter_and_optimizer']:raise ValueError('checkpoint recovery failed')
 if len({x['adapter_hash'] for x in state['runs']})!=1 or len({x['optimizer_hash'] for x in state['runs']})!=1:raise ValueError('parameter/optimizer drift')
 meta=json.loads((root/'evidence/stage28_measured/full_checkpoints_manifest.json').read_text())
 if len(meta['all_checkpoints'])!=10:raise ValueError('10 checkpoint manifests missing')
 for path in meta['published_representatives']:
  reference=next(item for item in meta['all_checkpoints'] if item['path']==path)
  if file_sha(root/'evidence/stage28_measured/checkpoints'/path)!=reference['sha256']:raise ValueError('checkpoint sample sha mismatch')
 output={'status':'research_artifact_evidence_pass_not_production_approved',
  'workstreams':{
   'stage25_8b_serving':{'status':'pass','verified_offered':serving['offered_requests'],'valid_cases':serving['cases'],
     'shared_prefix_cache_relative_tps_gain':serving['factor_averages']['prefix_cache_shared']['mean_relative_useful_tps_change'],
     'caveat':'Engineered shared prefix; not production traffic'},
   'stage26_real_sft':{'status':'pass','five_training_trials':5,'licensed_frozen_test':sft['test']['heldout_records'],
     'selected_rank16_test_ce':sft['test']['selected_ce'],'untuned_base_test_ce':sft['test']['base_ce'],
     'caveat':'Short 192-exposure training, limited reference quality evaluation'},
   'stage27_qwen3_shapes':{'status':'pass','correctness_checked_gpu_cases':len(kernel['cases']),
     'fused_bf16_256_vs_inductor_speedup':next(x['speedup_vs_compile'] for x in kernel['cases'] if x['name']=='fused_rms_rows256_bfloat16'),
     'caveat':'Microbenchmarks; not end-to-end model replacement'},
   'stage28_async_checkpoint':{'status':'pass','training_modes':4,'saved_checkpoints_verified':10,
     'exact_optimizer_and_adapters':True,'caveat':'No throughput win guaranteed; single-GPU'}},
  'open_critical_gates':['Production traffic generalization, longer multi-seed uncertainty intervals, cold/warm cache hit verification',
    'SFT task-level semantic/human preference and longer complete-data training','Kernel end-to-end transformer integration and Nsight Compute hardware counters',
    'Multi-GPU genuine FSDP FULL_SHARD and distributed checkpoint scaling unavailable on current single GPU'],
  'excluded_invalid_experiments':['stage25_invalid_v1 detailed-output length failed','stage26_invalid_v1 microbatch2 CUDA OOM',
    'stage27_invalid_v1 Triton timing API mismatch','stage29_invalid_v1 shell syntax failure','stage29_invalid_v2 missing attention_mask interrupted'],
  'note':'This verifies evidence contracts, not production SLO or research significance.'}
 if (root/'evidence/stage29_measured/quality.json').exists():
  q=json.loads((root/'evidence/stage29_measured/quality.json').read_text())
  if q['total_cases']!=96 or q['status']!='measured':raise ValueError('incomplete 96 case generation')
  output['workstreams']['stage29_generative_quality']={'status':'pass','paired_cases':96,
        'quality_proxy_ROUGE_L_change':q['overall']['paired_diff']['rouge_l_f1'],
        'caveat':'Lexical reference overlap, not human semantic quality'}
 else:output['workstreams']['stage29_generative_quality']={'status':'pending_separate_gpu_evaluation'}
 return output

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path.cwd());p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 x=audit(a.root)
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(x,indent=2)+'\n')
 print('FOUR_TRACK_AUDIT_PASS',x['status']);print(json.dumps(x['workstreams'],indent=2))
if __name__=='__main__':main()
