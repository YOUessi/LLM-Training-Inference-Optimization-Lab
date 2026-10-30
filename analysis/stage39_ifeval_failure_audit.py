"""Post-hoc cause investigation, not a holdout performance estimator.

Analyze ALL Stage36 IFEval exact paired outcomes and generated text, including
cap hits and repetitiveness; do not choose future model settings on these 96.
"""
from __future__ import annotations
import argparse,collections,json,math,random,re,statistics
from pathlib import Path
MODELS=('base','stage26_short','stage31_full')

def lexical_distinct2(text:str)->float|None:
 tokens=re.findall(r'\w+',text.lower(),flags=re.UNICODE)
 if len(tokens)<15:return None
 pairs=list(zip(tokens,tokens[1:]))
 return len(set(pairs))/len(pairs)

def exact_mcnemar(a_only:int,b_only:int):
 from math import comb
 n=a_only+b_only
 if n==0:return 1.
 lower=min(a_only,b_only)
 p=min(1.,2.*sum(comb(n,k) for k in range(lower+1))/(2**n))
 return p

def audit(root:Path)->dict:
 v=json.loads((root/'official_verifier_scores.json').read_text())
 g=json.loads((root/'generations.json').read_text())
 if v['status']!=g['status']!='measured':raise ValueError('invalid official grader status')
 if len(v['per_prompt'])!=len(g['cases'])!=96:raise ValueError('expected 96 matched official prompts')
 bykey={x['key']:x for x in g['cases']}
 if len(bykey)!=96:raise ValueError('duplicate generation keys')
 pairs=[]
 for item in v['per_prompt']:
  raw=bykey[item['key']]
  models={}
  for model in MODELS:
   m=item['metrics'][model];text=raw['responses'][model]
   if m['output_tokens']!=raw['output_token_counts'][model]:raise ValueError('length mismatch')
   models[model]={'strict':m['strict'],'loose':m['loose'],'at384':m['hit_cap'],
    'response_tokens':m['output_tokens'],'distinct2':lexical_distinct2(text)}
  pairs.append({'key':item['key'],'tier':item['primary_tier'],'models':models})
 comparisons={}
 for other in ('stage26_short','stage31_full'):
  a_only=sum(x['models']['base']['strict'] and not x['models'][other]['strict'] for x in pairs)
  b_only=sum(x['models'][other]['strict'] and not x['models']['base']['strict'] for x in pairs)
  summary={'base_only_correct':a_only,'other_only_correct':b_only,'both_correct':sum(x['models'][other]['strict'] and x['models']['base']['strict'] for x in pairs),
    'both_wrong':sum(not x['models'][other]['strict'] and not x['models']['base']['strict'] for x in pairs),
    'exact_paired_McNemar_twosided':exact_mcnemar(a_only,b_only)}
  comparisons['base_vs_'+other]=summary
 info={}
 for model in MODELS:
  rows=[x['models'][model] for x in pairs]
  capped=[r for r in rows if r['at384']];uncapped=[r for r in rows if not r['at384']]
  distinct=[r['distinct2'] for r in rows if r['distinct2'] is not None]
  info[model]={'strict_count':sum(r['strict'] for r in rows),'capped_count':len(capped),
    'strict_accuracy_among_capped':statistics.mean(r['strict'] for r in capped) if capped else None,
    'strict_accuracy_among_uncapped':statistics.mean(r['strict'] for r in uncapped) if uncapped else None,
    'mean_output_tokens':statistics.mean(r['response_tokens'] for r in rows),
    'mean_distinct_word_bigrams':statistics.mean(distinct),
    'severely_repetitive_under_0_25_distinct2_count':sum(r['distinct2'] is not None and r['distinct2']<.25 for r in rows)}
 families={}
 for family in sorted({x['tier'] for x in pairs}):
  subset=[x for x in pairs if x['tier']==family]
  families[family]={'n':len(subset),'strict_success':{m:sum(x['models'][m]['strict'] for x in subset) for m in MODELS},
    'hit384':{m:sum(x['models'][m]['at384'] for x in subset) for m in MODELS}}
 return {'status':'verified_diagnostic_not_holdout','kind':'stage39_stage36_repeat_truncation_ifeval_failure_audit',
  'n':96,'base_comparisons':comparisons,'model_stats':info,'by_tier':families,
  'limitations':['Post-hoc on already observed Stage36 prompts, must not tune on these and claim independent validation',
     'Cap vs non-cap strata are model-selected; uncapped conditional accuracy is not causal',
     'Word-level distinct-2 proxy is language-dependent and not a semantic factuality metric',
     'Paired McNemar tests on frozen sub-sample measure association, not training mechanism']}

def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=Path('evidence/stage36_official_ifeval'));p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 x=audit(a.source);a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(x,indent=2)+'\n')
 print('STAGE39_DIAGNOSTIC_PASS',x['n']);print(json.dumps(x['model_stats'],indent=2));print('PAIRED',x['base_comparisons'])
if __name__=='__main__':main()
