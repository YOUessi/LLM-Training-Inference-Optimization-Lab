"""Archive real IFBench60 official metrics, never redistribute private crowd responses."""
from __future__ import annotations
import argparse,collections,hashlib,json,pathlib,shutil

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--source',type=pathlib.Path,required=True)
 p.add_argument('--private-generated',type=pathlib.Path,required=True)
 p.add_argument('--original-generation-source-sha',type=pathlib.Path,required=True)
 p.add_argument('--out',type=pathlib.Path,default=pathlib.Path('evidence/stage44_external_ifbench'))
 a=p.parse_args()
 score=json.loads((a.source/'official_scores_public.json').read_text())
 private=json.loads(a.private_generated.read_text())
 if score['status']!='measured' or private['status']!='measured':raise ValueError('stage44 missing real work')
 if not(score['count']==score['verifier_count']==len(score['public_anonymized_per_prompt_checks'])==60):
  raise ValueError('60/60 official IFBench verifier outcomes missing')
 if not(len(private['raw_outputs'])==60 and score['original_generation_sha256']==hashlib.sha256(a.private_generated.read_bytes()).hexdigest()):
  raise ValueError('private original generated output source sha changed')
 manifest=json.loads(pathlib.Path('data/stage44_ifbench_manifest.json').read_text())
 if score['selected_input_sha256']!=manifest['selected_sha256']:
  raise ValueError('original IFBench source sample mismatch')
 x=score['aggregated'];names=('base','lora_sft','request_router')
 if set(x)!=set(names):raise ValueError('missing a benchmark model')
 for name in names:
  if not 0<=x[name]['strict_correct']<=60:raise ValueError('invalid official pass count')
  if x[name]['strict_correct']!=sum(z['checks'][name]['strict'] for z in score['public_anonymized_per_prompt_checks']):
   raise ValueError('strict accuracy claim differs from raw official scores')
 a.out.mkdir(exist_ok=True,parents=True)
 for name in ('official_scores_public.json','source_sha.txt'):
  file=a.source/name
  if file.is_file():shutil.copy2(file,a.out/name)
 local={'private_original_generated_results_path':str(a.source/'local_private_full_generations.json'),
  'original_private_generation_sha256':hashlib.sha256(a.private_generated.read_bytes()).hexdigest(),
  'original_generation_source_sha':a.original_generation_source_sha.read_text().strip(),
  'corrected_official_score_source_sha':(a.source/'source_sha.txt').read_text().strip(),
  'selected_original_IFBench60_source_sha256':manifest['selected_sha256'],
  'private_original_human_source_file':'results/stage44_ifbench_source/stage44_ifbench_external60.jsonl',
  'original_human_messages_not_committed_for_privacy':True,
  'note':'Source recipe and data SHA allow legal reconstruction. Human prompt text and full responses remain local.'}
 (a.out/'local_private_generation_provenance.json').write_text(json.dumps(local,indent=2)+'\n')
 (a.out/'SHA256SUMS').write_text(''.join(hashlib.sha256(f.read_bytes()).hexdigest()+'  '+f.name+'\n'
  for f in sorted(a.out.iterdir()) if f.is_file() and f.name!='SHA256SUMS'))
 lines=['# Stage44 measured external IFBench60 — independent OOD check of Stage42 frozen router',
  '', '**Real Qwen3-8B NF4 GPU experiment.** Official AllenAI IFBench 60 SHA-frozen test inputs, seven **non-classic IFEval** first-level precision constraint families, scored via unchanged IFBench 83-verifier package from pinned `1c40f0c` commit. Actual 60 prompt × two 8B model conditions = **120 real GPU generations** (non-thinking greedy max384 tokens), with the third routed condition choosing one of those two outputs by Stage42 fixed request-only regex **before any official scoring**. The public benchmark results below are from the original official strict and loose evaluators, not self-grading or simulated outcomes. Full private crowd user prompt and generated answer text stays on authorized Tang to avoid gratuitous republishing; GitHub retains per-prompt hashes, route decisions, full official booleans, source SHA, source code, and aggregate metrics under `evidence/stage44_external_ifbench/`.',
  '', '| Frozen model condition | Strict all-constraint prompts /60 | Strict % | Loose % | Mean output tokens | Reached 384-token limit |',
  '|---|---:|---:|---:|---:|---:|']
 for model in names:
  row=x[model]
  lines.append(f"| {model} | {row['strict_correct']}/60 | {row['strict_prompt_accuracy']*100:.2f}% | {row['loose_prompt_accuracy']*100:.2f}% | {row['mean_output_tokens']:.1f} | {row['hit_output_cap_fraction']*100:.1f}% |")
 lines+=['', '## Frozen out-of-domain acceptance',
  '', '| Stage44 preregistered criterion | Actual result |', '|---|---|']
 for key,flag in score['preregistered_generalization_acceptance_checks'].items():
  lines.append(f"| {key} | {'PASS' if flag else 'FAIL'} |")
 lines+=['',f"**Overall IFBench out-of-domain transfer gate: {'PASS' if score['cross_benchmark_go'] else 'FAIL'}.** This is the first external constraint-family test after Stage42's IFEval/Dolly joint research gate; we did **not** change router weights, threshold, samples or LoRA after seeing IFBench outcomes. A failure means the Stage42 result cannot be claimed to generalize across instruction-following benchmarks.",
 '', '## Matched per-family outcomes',
  '', '| IFBench first instruction family | Original base strict passes | Unscaled full SFT | Router |', '|---|---:|---:|---:|']
 for name,rows in sorted(score['first_constraint_family_pass_counts'].items()):
  lines.append(f"| {name} | {rows['base']} | {rows['lora_sft']} | {rows['request_router']} |")
 lines+=['', '## Within-test family-stratified paired bootstrap, exploratory',
  '', '| Comparator | Router minus comparator strict success interval |','|---|---:|']
 for name,ci in score['matched_3000_stratified_bootstrap_router_minus'].items():
  lines.append(f'| {name} | [{ci[0]*100:+.2f}, {ci[1]*100:+.2f}] percentage points |')
 lines+=['', '## Scope and failure modes',
 '', 'The test is a preselected **60/300** subset, not official full IFBench leaderboard; fixed quota sampling is not a sample of natural production traffic. Original IFBench may contain third-party model-derived data and WildChat user content under its ODC-BY-1.0 dataset licence and specific model/responsible-use terms; raw question and response text is retained locally only. The official IFBench verifier evaluates precise constraint compliance (format, words, ratios, etc.), not human factuality/safety/alignment. This experiment uses Qwen3 NF4 BF16 and cannot automatically transfer to the AWQ FP16 vLLM deployment tested in Stage43. Most importantly, a 384-token output cap may artificially lower strict success for long-output tasks, so cap-hit fractions are required in every interpretation. Only one LoRA model and 60 external examples; seed-level model training generalization is untested.',
 '', 'Reproducible sources: `docs/STAGE44_PROTOCOL.md`, `data/stage44_ifbench_manifest.json`, official vendored `third_party/ifbench/`, `src/llm_opt_lab/training/stage44_external_ifbench.py`, and numeric score `evidence/stage44_external_ifbench/official_scores_public.json`.']
 pathlib.Path('docs/STAGE44_MEASURED_RESULTS.md').write_text('\n'.join(lines)+'\n')
 print('STAGE44_IFBENCH_REAL_EVIDENCE_ARCHIVED',score['cross_benchmark_go'],x,flush=True)
if __name__=='__main__':main()
