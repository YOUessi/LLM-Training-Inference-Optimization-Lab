"""Release 120 real AWQ-vLLM IFBench calls vs earlier frozen NF4 result.

Never distribute original crowd prompt/response text. The 60 original source
IDs must match exactly; no post-hoc model/router selection is allowed.
"""
from __future__ import annotations
import argparse,hashlib,json,math,pathlib,shutil,statistics

def load(path):return json.loads(path.read_text())
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--source',type=pathlib.Path,required=True)
 ap.add_argument('--original-nf4',type=pathlib.Path,default=pathlib.Path('evidence/stage44_external_ifbench/official_scores_public.json'))
 ap.add_argument('--out',type=pathlib.Path,default=pathlib.Path('evidence/stage46_awq_ifbench_quality'))
 a=ap.parse_args()
 x=load(a.source/'official_scores.json');nf4=load(a.original_nf4)
 raw=a.source/'private_generation.json';gen=load(raw)
 if not(x['status']==nf4['status']==gen['status']=='measured' and x['count']==nf4['count']==60):
  raise ValueError('both NF4/AWQ quality tests must have 60 official measured samples')
 if x['kind']!='stage46_official_IFBench60_AWQ_vLLM_base_plus_LoRA' or gen['kind']!='stage46_awq_vllm_dual_ifbench_external_60_real_response_text':
  raise ValueError('stage46 not a verified real AWQ API model generation')
 if x['original_generation_sha256']!=hashlib.sha256(raw.read_bytes()).hexdigest():
  raise ValueError('AWQ private actual-generation output SHA mismatch')
 if x['selected_input_sha256']!=nf4['selected_input_sha256']:
  raise ValueError('not identical original IFBench source cohort')
 if set(gen['registered_model_ids'])!={'qwen3-8b-awq-local','qwen3-8b-dolly-r16'}:
  raise ValueError('the dual serving model IDs were not registered')
 if not x['reused_unmodified_official_ifbench_scorer']:raise ValueError('unverified official comparator')
 orig={r['key']:r for r in nf4['public_anonymized_per_prompt_checks']}
 actual={r['key']:r for r in x['public_anonymized_per_prompt_checks']}
 if len(orig)!=len(actual)!=60 or set(orig)!=set(actual):raise ValueError('source key identity changed between environments')
 roles=('base','lora_sft','request_router')
 pair=[]
 for key,before in orig.items():
  after=actual[key]
  if before['prompt_sha256']!=after['prompt_sha256'] or before['families']!=after['families'] or before['route']!=after['route']:
   raise ValueError('original prompt or routed choice changed across backend env')
  diff={'key':key,'tier':before['families'][0],'routed_model':before['route']}
  for role in roles:
   v0=before['checks'][role];v1=after['checks'][role]
   diff[role]={'nf4_strict':v0['strict'],'awq_strict':v1['strict'],
    'nf4_loose':v0['loose'],'awq_loose':v1['loose'],
    'nf4_cap':v0['hit_cap'],'awq_cap':v1['hit_cap'],
    'nf4_output_tokens':v0['output_tokens'],'awq_output_tokens':v1['output_tokens']}
  pair.append(diff)
 comparisons={}
 for role in roles:
  allpair=[z[role] for z in pair]
  nf4only=sum(v['nf4_strict'] and not v['awq_strict'] for v in allpair)
  awqonly=sum(v['awq_strict'] and not v['nf4_strict'] for v in allpair)
  both=sum(v['nf4_strict'] and v['awq_strict'] for v in allpair)
  neither=sum(not v['nf4_strict'] and not v['awq_strict'] for v in allpair)
  if nf4only+awqonly+both+neither!=60:raise ValueError('invalid paired discordance')
  if sum(v['nf4_strict'] for v in allpair)!=nf4['aggregated'][role]['strict_correct']:
   raise ValueError('NF4 aggregate output changed')
  if sum(v['awq_strict'] for v in allpair)!=x['aggregated'][role]['strict_correct']:
   raise ValueError('AWQ aggregate output changed')
  comparisons[role]={'nf4_strict_passes':sum(v['nf4_strict'] for v in allpair),
   'awq_strict_passes':sum(v['awq_strict'] for v in allpair),
   'nf4_only_correct':nf4only,'awq_only_correct':awqonly,'both_correct':both,'both_wrong':neither,
   'strict_accuracy_delta_awq_minus_nf4_pp':(awqonly-nf4only)/60*100,
   'nf4_output_cap_fraction':statistics.mean(v['nf4_cap'] for v in allpair),
   'awq_output_cap_fraction':statistics.mean(v['awq_cap'] for v in allpair),
   'mean_nf4_output_tokens':statistics.mean(v['nf4_output_tokens'] for v in allpair),
   'mean_awq_output_tokens':statistics.mean(v['awq_output_tokens'] for v in allpair)}
 summary={'status':'verified','kind':'stage46_AWQ_actual_vllm_vs_NF4_offline_IFBench60_same_source_paired',
  'original_ifbench_selected_sha256':x['selected_input_sha256'],
  'awq_source_git_sha':(a.source/'source_sha.txt').read_text().strip(),
  'nf4_official_score_sha256':hashlib.sha256(a.original_nf4.read_bytes()).hexdigest(),
  'awq_official_score_sha256':hashlib.sha256((a.source/'official_scores.json').read_bytes()).hexdigest(),
  'awq_private_real_api_generations_sha256':hashlib.sha256(raw.read_bytes()).hexdigest(),
  'real_awq_api_calls':120,'original_ifbench_prompts':60,
  'observed_model_registry':gen['registered_model_ids'],
  'paired_roles':comparisons,
  'nf4_original_cross_benchmark_gate':nf4['cross_benchmark_go'],
  'awq_actual_server_cross_benchmark_gate':x['cross_benchmark_go'],
  'conditioned_not_production_approval':True,
  'per_prompt_comparison_no_text':pair,
  'limitations':['Same IFBench60 heldout cohort as Stage44 NF4: comparison measures deployment differences, not new independent validation',
   'AWQ vs NF4 base and LoRA are different quantization/compute regimes, model responses may vary',
   'Official IFBench checks constraint obedience, not semantic user satisfaction or safety',
   'The public IFBench source may include third party WildChat crowd content: keep prompts and response strings locally private',
   'Two model counters generated sequentially in actual vLLM; Stage45 measures concurrent routed server performance separately',
   '384 output token cap potentially truncates natural responses, original source test has only 60 cases']}
 a.out.mkdir(parents=True,exist_ok=True)
 shutil.copy2(a.source/'official_scores.json',a.out/'official_scores_public.json')
 shutil.copy2(a.source/'source_sha.txt',a.out/'source_sha.txt')
 for f in ('allocated_gpu.csv','score.log'):
  path=a.source/f
  if path.is_file():shutil.copy2(path,a.out/f)
 (a.out/'paired_AWQ_vs_NF4.json').write_text(json.dumps(summary,indent=2)+'\n')
 meta={'private_original_answer_source':str(raw),
  'real_API_gen_sha256':hashlib.sha256(raw.read_bytes()).hexdigest(),
  'only_original_benchmark_input_hashes_and_official_scores_uploaded':True,
  'private_generation_and_full_server_logs_remain_on_tang':True}
 (a.out/'private_file_provenance.json').write_text(json.dumps(meta,indent=2)+'\n')
 (a.out/'SHA256SUMS').write_text(''.join(hashlib.sha256(f.read_bytes()).hexdigest()+'  '+f.name+'\n'
   for f in sorted(a.out.glob('*')) if f.is_file() and f.name!='SHA256SUMS'))
 lines=['# Stage46: 120 actual AWQ vLLM API responses vs NF4/BF16 IFBench60 offline quality',
  '', '**Measured:** On actual Tang vLLM 0.19.1 dual model server (loopback; AWQ/FP16 base and the NF4-trained rank16 LoRA), recorded **60 frozen IFBench prompts ×2 actual HTTP responses = 120 vLLM API calls**, deterministic thinking-disabled greedy max384, same unmodified official AllenAI IFBench strict/loose verifier from pinned `1c40f0c`. The router selected a model using only original prompt *before* generated responses; this experiment then evaluated both candidate answers for a paired counterfactual, not a three-way network batch. Raw human prompt / generated response strings remain on private Tang and only anonymized original cohort IDs/response hashes and official verdicts are committed.',
  '', '## Exact paired Stage44 NF4 vs Stage46 AWQ, same original 60 IFBench source keys',
  '', '| Model | NF4 strict passed | AWQ actual vLLM strict passed | Δ AWQ−NF4 | NF4-only correct | AWQ-only correct | AWQ answer cap hits |',
  '|---|---:|---:|---:|---:|---:|---:|']
 for role,v in comparisons.items():
  lines.append(f"| {role} | {v['nf4_strict_passes']}/60 | {v['awq_strict_passes']}/60 | {v['strict_accuracy_delta_awq_minus_nf4_pp']:+.2f} pp | {v['nf4_only_correct']} | {v['awq_only_correct']} | {v['awq_output_cap_fraction']*100:.1f}% |")
 lines+=['', f"**Pre-existing IFBench relative generalization gate:** NF4 offline Stage44 {'PASS' if nf4['cross_benchmark_go'] else 'FAIL'}; actual AWQ vLLM Stage46 {'PASS' if x['cross_benchmark_go'] else 'FAIL'}. These are preregistered within-benchmark relative criteria, **not** minimum absolute accuracy or deployment success; 60 new constraint prompts of seven families do not test all user/task populations.",
  '', '## Interpretation and limitations',
  '', 'Stage44 NF4 and Stage46 AWQ are deliberately tested on the **same** exact IFBench60 texts/constraints, allowing direct paired quantization/backend compatibility verification; their agreement is not evidence of another independent new dataset. Different decoding responses and cap hits are real quantization/backend effects and need further investigation. Stage45 separately measured 864 concurrent API responses under actual dual vLLM but was primarily a serving latency/load experiment on multilingual OASST1 source, not a correctness checker. Strict IFBench is still low absolute, output caps remain material. No proof of human-rated helpfulness, safe complete answers, hour-scale traffic performance, real customer task performance, or multi-GPU training. Production rollout BLOCKED.',
  '', 'Source: `docs/STAGE46_PROTOCOL.md`, `scripts/run_stage46_awq_ifbench.sh`, `src/llm_opt_lab/serving/stage46_vllm_ifeval_quality.py`, `evidence/stage46_awq_ifbench_quality/` and original NF4 `evidence/stage44_external_ifbench/`.']
 pathlib.Path('docs/STAGE46_MEASURED_RESULTS.md').write_text('\n'.join(lines)+'\n')
 print('STAGE46_REAL_VLLM_120_AND_NF4_PAIR_AUDIT',comparisons,'AWQ_GATE',x['cross_benchmark_go'])
if __name__=='__main__':main()
