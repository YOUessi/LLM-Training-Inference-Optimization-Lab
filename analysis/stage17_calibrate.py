"""Fit robust mode-specific response-time priors on one seed, evaluate other seed.
Diagnostic target is client completion latency, not pure model service time.
"""
import json,pathlib,statistics,sys
root=pathlib.Path('evidence/stage15_multitrack')
prior={'label':55.49,'detailed':1077.63}
data=[]
for p in sorted(root.glob('*.json')):
 d=json.loads(p.read_text())
 for r in d['requests']:
  if r['status']=='ok':data.append({'mode':r['output_mode'],'latency':r['latency_ms'],'domain':d['config']['domain'],'rate':d['config']['rate_per_second'],'policy':d['config']['policy'],'pred':r['policy_decision']['predicted_service_ms'],'id':r['request_id']})
# Group-disjoint by source domain: k8s calibration, prose evaluation.
train=[r for r in data if r['domain']=='k8s' and r['policy']=='fifo_static400']
test=[r for r in data if r['domain']=='prose' and r['policy']=='fifo_static400']
fit={m:statistics.median(r['latency'] for r in train if r['mode']==m) for m in prior}
metrics={}
for mode in prior:
 a=[r for r in test if r['mode']==mode]
 errors_before=[abs(r['latency']-r['pred']) for r in a]
 errors_after=[abs(r['latency']-fit[mode]) for r in a]
 metrics[mode]={'train_n':sum(r['mode']==mode for r in train),'test_n':len(a),'median_fit_ms':fit[mode],
 'test_mae_baseline_ms':statistics.mean(errors_before),'test_mae_fitted_ms':statistics.mean(errors_after),
 'test_median_absolute_error_baseline_ms':statistics.median(errors_before),
 'test_median_absolute_error_fitted_ms':statistics.median(errors_after)}
out={'status':'retrospective_cross_domain_diagnostic','training_source':'k8s FIFO only','evaluation_source':'prose FIFO only','metrics':metrics,'limitations':['Latency is total client request time rather than model service time','Domains reused in prior development experiments','Prediction after fit is constant by output mode; not yet an online scheduler model','No claim of real scheduling utility improvement']}
p=pathlib.Path('results/stage17_calibration.json');p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(metrics,indent=2))
