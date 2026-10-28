"""Fail-closed original utility gate against actual unlimited baseline."""
import argparse,json,pathlib
p=argparse.ArgumentParser();p.add_argument('--stage22',type=pathlib.Path,required=True);p.add_argument('--stage23',type=pathlib.Path,required=True);p.add_argument('--out',type=pathlib.Path,required=True);a=p.parse_args()
refs={}
for path in sorted(a.stage23.glob('stdlib_seed*_rate*_unlimited.json')):
 d=json.loads(path.read_text());c=d['config'];s=d['summary']
 assert d['status']=='measured' and s['completed']==s['offered']==24 and s['rejected']==0 and s['errors']==0
 refs[(c['seed'],c['rate_per_second'])]=d
assert len(refs)==4
rows=[]
for path in sorted(a.stage22.glob('stdlib_seed*.json')):
 d=json.loads(path.read_text());c=d['config'];s=d['summary'];key=(c['seed'],c['rate_per_second']);base=refs[key];b=base['summary']
 assert d['status']=='measured' and s['offered']==24 and s['completed']+s['rejected']==24 and s['errors']==0
 assert c['arrival_schedule_offsets_s']==base['config']['arrival_schedule_offsets_s']
 assert [x['request_id'] for x in d['requests']]==[x['request_id'] for x in base['requests']]
 assert c['frozen_requests_sha256']==base['config']['frozen_requests_sha256']
 base_tps=b['useful_output_tokens_per_second'];tps=s['useful_output_tokens_per_second']
 baseline=b['slo_all_offered']['250']['compliance_fraction']+.20
 score=s['slo_all_offered']['250']['compliance_fraction']-.50*s['rejected']/24+.20*tps/base_tps
 delta=score-baseline
 good=(delta>=.02 and tps>=.8*base_tps and s['rejected']<=4)
 rows.append({'seed':c['seed'],'rate_per_second':c['rate_per_second'],'policy':c['policy'],'offered':24,
    'completed':s['completed'],'rejected':s['rejected'],
    'candidate_slo250':s['slo_all_offered']['250']['compliance_fraction'],
    'unlimited_slo250':b['slo_all_offered']['250']['compliance_fraction'],
    'candidate_useful_tps':tps,'unlimited_useful_tps':base_tps,
    'candidate_utility':score,'unlimited_utility':baseline,'utility_delta':delta,
    'passes_original_guardrails':good})
assert len(rows)==16
report={'schema':'stage23-unlimited-real-request-paired-gate-v1','status':'pass','reference_arms':4,'candidate_arms':16,'all_offered':480,
 'candidate_wins':sum(x['passes_original_guardrails'] for x in rows),'candidate_rows':rows,
 'limitations':['The accepted strategy runs and unlimited reference are from separate GPU sessions, not fully randomized order','Small 24-offer cells; 2 arrival seeds and one synthetic-user-question source domain','Original candidate guardrails apply independently per matched cell, not retrospectively averaged']}
a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2)+'\n')
print('STAGE23_STRICT_VALIDATION_PASS candidate_wins',report['candidate_wins'],'of',len(rows))
for x in rows:
 print(x['seed'],x['rate_per_second'],x['policy'],'rejects',x['rejected'],'delta_U',round(x['utility_delta'],4),'pass',x['passes_original_guardrails'])
