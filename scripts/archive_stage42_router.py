"""Verified offline-router research report from real Qwen3 base/LoRA GPU evidence."""
from __future__ import annotations
import argparse,hashlib,json,pathlib,shutil


def archive(source:pathlib.Path,out:pathlib.Path):
 payload=json.loads((source/'frozen_router_result.json').read_text())
 if payload['status']!='measured' or payload['kind']!='stage42_prompt_only_base_vs_sft_constraint_router_clean_96_128':
  raise ValueError('Stage42 real run not yet measured')
 if not(len(payload['real_ifeval_generations'])==len(payload['real_ifeval_official_verdicts'])==96 and
        len(payload['real_dolly_reference_ce'])==128):raise ValueError('missing result rows')
 if not payload['router']['uses_official_labels_or_model_responses'] is False:raise ValueError('hidden label leak')
 source_sha=(source/'source_sha.txt').read_text().strip()
 if len(source_sha)!=40:raise ValueError('source git SHA missing')
 if payload['previous_source_identites_excluded']!={'IFEval':325,'Dolly':960}:
  raise ValueError('dataset frozen exclusion failed')
 if not all(isinstance(v,bool) for v in payload['predeclared_joint_research_gate'].values()):
  raise ValueError('invalid accepted joint gate')
 if payload['joint_gate_pass']!=all(payload['predeclared_joint_research_gate'].values()):
  raise ValueError('gate calculation inconsistent')
 models=('base','lora_sft','request_router')
 actual={model:sum(x[model]['strict'] for x in payload['real_ifeval_official_verdicts']) for model in models}
 if any(actual[m]!=payload['official_ifeval_strict'][m]['correct'] for m in models):
  raise ValueError('official grader pass-count drift')
 for metric in payload['real_dolly_reference_ce']:
  if set(metric['CE'])!=set(models):raise ValueError('missing CE counterfactual')
  if metric['CE']['request_router']!=metric['CE'][metric['route']]:raise ValueError('router CE not selected before score')
 out.mkdir(exist_ok=True,parents=True)
 for file in ('frozen_router_result.json','source_sha.txt','generation.log'):
  p=source/file
  if p.is_file():shutil.copy2(p,out/file)
 (out/'SHA256SUMS').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.name+'\n' for p in sorted(out.glob('*')) if p.is_file() and p.name!='SHA256SUMS'))
 reported=payload['official_ifeval_strict'];ce=payload['dolly_reference_answer_ce'];decision=payload['predeclared_joint_research_gate']
 text=['# Stage42 — prompt-only hard-constraint routing, new IFEval96 and Dolly128 ID-disjoint tests',
  '', '**Real measured on Tang RTX4090 Laptop 16GiB**: original Qwen3-8B NF4 vs frozen Stage31 full512 rank16 QLoRA vs a lightweight deterministic prompt-only rule switching to the original model on explicit format/count/negative/structural constraints. This is a **handcrafted inference-time specialist router, not a new trained model**. The request decision uses *only* user prompt text and frozen regex weights, no official benchmark label, reference answer or generated text. For auditing, both original and LoRA counterfactual responses are generated and scored offline before routing metrics are computed, but the actual router requires only one model forward per request. Exactly 96 **new** IFEval source keys excluding 325 previously observed ones, and 128 **new** human-written Dolly source identities excluding previous 960 selected items, all frozen before GPU run. Original source SHA, licensed source, per-case model output SHA, official Google rule scores and full answer-only CE are in `evidence/stage42_router/`.',
  '', '## Previously unseen Google IFEval96 subset: official strict instruction compliance',
  '', '| Frozen model choice | Strict passed /96 | Strict prompt accuracy | Loose prompt accuracy |',
  '|---|---:|---:|---:|']
 for model in models:
  v=reported[model]
  text.append(f"| {model} | {v['correct']}/96 | {v['strict_prompt_fraction']*100:.2f}% | {v['loose_prompt_fraction']*100:.2f}% |")
 text+=['','## New Dolly128 reference answer-token CE (same test IDs and target tokens)',
  '', '| Frozen model choice | CE (nats / target token) | Perplexity |', '|---|---:|---:|']
 for model in models:
  v=ce[model]
  text.append(f"| {model} | {v['ce_nats']:.5f} | {v['perplexity']:.4f} |")
 text+=['', '## Router trace and research go/no-go',
  f"- IFEval route counts: {payload['route_counts']['ifeval']} (hard-constraint priority to base).",
  f"- Dolly route counts: {payload['route_counts']['dolly']} (ordinary request priority to LoRA).",
  f"- Predeclared joint research gate: **{'PASS' if payload['joint_gate_pass'] else 'FAIL'}**; conditions {decision}. Gate requires official IFEval router successes within 2/96 of original base AND Dolly128 answer CE at least 0.50 nats/token better than original base, with no official rule-checker output as routing input.",
  '', '## Within-source stratified case bootstrap (exploratory only)',
  '', '| Comparison | IFEval strict accuracy delta (router minus comparator) | Dolly answer CE delta (router minus comparator) |',
  '|---|---:|---:|']
 for comp in ('base','lora_sft'):
  a=payload['paired_bootstrap_uncertainty']['ifeval_router_minus'][comp]
  b=payload['paired_bootstrap_uncertainty']['dolly_router_ce_minus'][comp]
  text.append(f'| {comp} | [{100*a[0]:+.2f}, {100*a[1]:+.2f}] pp | [{b[0]:+.4f}, {b[1]:+.4f}] nats/token |')
 text+=['', '## Explicit limitations / non-production status',
  '', 'One pretrained model and one LoRA checkpoint, public IFEval and Dolly *within-source* previously-unseen identities (potential pretrained knowledge overlap), and a lexical constraint router whose correct classification on benchmark-format instructions may be trivial. This is an intervention evaluating a baseline *engineering risk-routing rule*, **not** a learned novel algorithm or proof it generalizes to implicit, long-context, multilingual user constraints. The baseline and LoRA counterfactuals were generated sequentially for fair score verification; true deployed model routing would call one backend only and needs a separate latency/SLO load study. IFEval strict checker measures programmatic constraint obedience, not factual helpfulness or safety. Conditional fixed-case bootstrap is not true multi-seed population inference. Production authorization remains blocked until genuine user quality/error rates, end-to-end vLLM dual-backend routing and larger varied tasks are verified.',
  '', 'Sources: `docs/STAGE42_PROTOCOL.md`, `src/llm_opt_lab/training/stage42_risk_router.py`, frozen `data/stage42_manifest.json`, machine source `evidence/stage42_router/frozen_router_result.json`.']
 pathlib.Path('docs/STAGE42_MEASURED_RESULTS.md').write_text('\n'.join(text)+'\n')
 print('STAGE42_RESEARCH_EVIDENCE_ARCHIVED',payload['joint_gate_pass'],'MODEL_OBEDIENCE',actual,'ROUTE',payload['route_counts'],flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source',type=pathlib.Path,required=True)
 p.add_argument('--out',type=pathlib.Path,default=pathlib.Path('evidence/stage42_router'));a=p.parse_args();archive(a.source,a.out)
