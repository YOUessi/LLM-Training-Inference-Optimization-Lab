"""Archive 128 record-disjoint Dolly CE experiment after Stage38/40 fixed decisions."""
from __future__ import annotations
import argparse,hashlib,json,pathlib,shutil

def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=pathlib.Path,required=True)
 p.add_argument('--out',type=pathlib.Path,default=pathlib.Path('evidence/stage41_new_dolly128'))
 a=p.parse_args()
 x=json.loads((a.source/'fresh128_four_conditions.json').read_text())
 if x['status']!='measured' or not x['excluded_previous832_identites']:
  raise ValueError('invalid original independent records')
 if x['counts']!={'examples':128,'categories':8,'models':4}:
  raise ValueError('incomplete four-model 128 CE test')
 if len(x['per_example_answer_ce'])!=4 or any(len(v)!=128 for v in x['per_example_answer_ce'].values()):
  raise ValueError('incomplete per-model CE rows')
 a.out.mkdir(parents=True,exist_ok=True)
 for name in ('fresh128_four_conditions.json','source_sha.txt','progress.log'):
  src=a.source/name
  if src.is_file():shutil.copy2(src,a.out/name)
 (a.out/'SHA256SUMS').write_text(''.join(hashlib.sha256(f.read_bytes()).hexdigest()+'  '+f.name+'\n'
  for f in sorted(a.out.glob('*')) if f.is_file() and f.name!='SHA256SUMS'))
 lines=['# Stage41: new 128 Dolly instructions never selected in any previous 832 source identities',
  '', '**Actual real Qwen3-8B NF4 GPU evaluation**. Freeze an independent-by-record CC BY-SA Databricks Dolly128 holdout (16 each from eight original categories), excluding all previously selected Stage26/31 train/val/test 704 and Stage34 separate128 (832 excluded) BEFORE the Stage38/Stage40 inference-time and format-SFT selection results. No new model/adapter/strength tuning based on this data. Source and selected decision SHA validated. Strictly identical assistant-answer target tokens / input messages for four preselected variants: unmodified base, old Stage31 rank16 SFT, Stage38 guarded residual α (may be α0 fallback), Stage40 constraint-format replay (may be base fallback). Real reference CE and perplexity only; not semantic factual correctness.',
 '', '| Pre-selected model | Untouched Dolly examples | Answer CE nats/token | PPL |',
 '|---|---:|---:|---:|']
 for name,score in x['overall'].items():
  lines.append(f"| {name} | {score['examples']} | {score['assistant_ce_nats']:.5f} | {score['perplexity']:.4f} |")
 lines+=['',f"**Previous dev choices (not from this test):** Stage38 residual α={x['selection38_dev_alpha']:.2f}; Stage40 format-training candidate `{x['selection40_dev_model']}`. Alpha=0 or selected='base' legitimately means no adapted model passed its guardrails; do **not** relabel such fallback as a successful new adapted model.",
 '', '## Stratified fixed-sample paired CE differences versus original base',
 '| Model | Within-source 95% bootstrap interval (CE new − original base) |','|---|---:|']
 for model,ci in x['paired_stratified_bootstrap_delta_ce_vs_original_base'].items():
  lines.append(f'| {model} | [{ci[0]:+.5f}, {ci[1]:+.5f}] nats/token |')
 lines+=['', '**Limitations:** The 128 sample identities are genuinely disjoint from previous 832 locally selected Dolly items, but Dolly is still the SAME publicly available upstream source and Qwen pretraining may have encountered it. Conditional bootstrap only resamples records and does not capture independent training seeds, models or natural human task distributions. Lower reference loss does not excuse a loss of strict IFEval instruction obedience (see Stage38/40 separate unseen tests) or certify safety/semantic correctness. Data license and source hashes are in `data/stage41_manifest.json`; full numeric per-case CE and SHA checks are in `evidence/stage41_new_dolly128/`.']
 pathlib.Path('docs/STAGE41_MEASURED_RESULTS.md').write_text('\n'.join(lines)+'\n')
 print('STAGE41_ARCHIVED_REAL128',x['overall'])
if __name__=='__main__':main()
