"""Archive the exact official Google IFEval strict/loose score; no manual grade override."""
from __future__ import annotations
import argparse,hashlib,json,pathlib,shutil

def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=pathlib.Path,required=True)
 p.add_argument('--out',type=pathlib.Path,default=pathlib.Path('evidence/stage36_official_ifeval'))
 a=p.parse_args()
 x=json.loads((a.source/'official_verifier_scores.json').read_text())
 g=json.loads((a.source/'generations.json').read_text())
 if x['status']!='measured' or len(x['per_prompt'])!=len(g['cases'])!=96:raise ValueError('incomplete official IFEval evaluation')
 if x['generator_json_sha256']!=hashlib.sha256((a.source/'generations.json').read_bytes()).hexdigest():
  raise ValueError('official scorer not bound to these model outputs')
 a.out.mkdir(parents=True,exist_ok=True)
 for f in ('official_verifier_scores.json','generations.json','source_sha.txt','generation.log','scoring.log'):
  src=a.source/f
  if src.is_file():shutil.copy2(src,a.out/f)
 (a.out/'SHA256SUMS').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.name+'\n'
  for p in sorted(a.out.glob('*')) if p.is_file() and p.name!='SHA256SUMS'))
 lines=['# Stage36: Qwen3-8B instruction-following on frozen Google IFEval official rule checker','',
  '**Actual measured:** exactly 96 preregistered prompts from the original 541 IFEval Apache-2.0 dataset (nine first-level instruction families, deterministic SHA ordering), three frozen Qwen3-8B NF4 models: base, Stage26 short rank16, Stage31 full512 rank16. Same original prompt, no thinking, greedy max384 output tokens, official Google `test_instruction_following_strict/loose` with original source pinned `7e672cc`. Per-prompt official booleans, generated text, SHA and throughput/response length archived in `evidence/stage36_official_ifeval/`. **Not the official full 541-prompt leaderboard result.**',
  '', '| Model | Strict all-instructions prompt accuracy | Loose prompt accuracy | Strict instruction accuracy | Loose instruction accuracy | Hit 384-token cap |',
  '|---|---:|---:|---:|---:|---:|']
 for model,metrics in x['overall'].items():
  lines.append('| '+model+' | '+' | '.join(f'{metrics[k]*100:.1f}%' for k in ('strict_prompt_accuracy','loose_prompt_accuracy','strict_instruction_accuracy','loose_instruction_accuracy','hit_output_cap_fraction'))+' |')
 lines+=['','## Paired conditional uncertainty (full512 SFT minus comparator)', '',
   '| Comparator | Strict prompt accuracy difference | Stratified prompt-level exploratory bootstrap 95% interval |',
   '|---|---:|---:|']
 for against,v in x['paired_stratified_uncertainty_full_vs'].items():
  ci=v['stratified_4000bootstrap_95pct_ci']
  lines.append(f"| {against} | {100*v['delta_strict_prompt_accuracy']:+.2f} pp | [{100*ci[0]:+.2f}, {100*ci[1]:+.2f}] pp |")
 lines+=['','## By original IFEval first-level instruction family (strict prompt accuracy)',
   '','| Tier | Base | Stage26 short | Stage31 full |','|---|---:|---:|---:|']
 for tier,r in sorted(x['by_primary_tier'].items()):
  lines.append('| '+tier+' | '+' | '.join(f'{r[m]*100:.1f}%' for m in ('base','stage26_short','stage31_full'))+' |')
 lines+=['','**Research caveats:** This is only 96 examples, sampled before model generation and with publicly frozen feasibility exclusions for known long-output/NTLK sentence resource constraints. Input response truncation at 384 tokens can make some published constraint requests impossible; cap-hit rates above must be considered. Official heuristic instruction compliance is stricter/more interpretable than the earlier independent NLI semantic proxy but still does **not** establish factual accuracy, safe helpfulness or human preference. Bootstrap CI conditions on fixed public source prompts and one trained model, not multi-seed model training. No post-hoc selection based on IFEval results.',
   '', 'Source: `docs/STAGE36_PROTOCOL.md`, frozen `data/stage36_ifeval_96.jsonl` & SHA manifest, unmodified official evaluator under `third_party/ifeval`, full generated outputs and scoring under `evidence/stage36_official_ifeval`.']
 pathlib.Path('docs/STAGE36_MEASURED_RESULTS.md').write_text('\n'.join(lines)+'\n')
 print('STAGE36_IFEVAL_ARCHIVED',len(x['per_prompt']),'cases',x['overall'])
if __name__=='__main__':main()
