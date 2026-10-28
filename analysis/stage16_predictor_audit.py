"""Audit historic model-specific scheduling predictions, no training leakage."""
import json,statistics,math,argparse,pathlib

def analyze(root):
    records=[]
    for p in sorted(root.glob('*.json')):
        d=json.loads(p.read_text())
        for r in d['requests']:
            if r['status']!='ok':continue
            predicted=r['policy_decision']['predicted_service_ms']
            observed=r['latency_ms']
            if predicted<=0 or observed is None:raise ValueError('missing valid observation')
            records.append(dict(mode=r['output_mode'],predicted=predicted,observed=observed,domain=d['config']['domain'],policy=d['config']['policy']))
    out={}
    for mode in ('label','detailed'):
        a=[r for r in records if r['mode']==mode]
        out[mode]={'n':len(a),'observed_median_ms':statistics.median(x['observed'] for x in a),
           'predicted_median_ms':statistics.median(x['predicted'] for x in a),
           'median_absolute_error_ms':statistics.median(abs(x['observed']-x['predicted']) for x in a),
           'median_observed_over_predicted':statistics.median(x['observed']/x['predicted'] for x in a)}
    return {'kind':'retrospective-in-sample-audit-not-validated-calibration','source':str(root),'completed':len(records),'by_mode':out,
            'limitations':'Measured client request latency includes engine time and overhead, not pure GPU service; repeated request identities and policy-adaptive predictions are non-independent; no model retraining has occurred.'}
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--input',type=pathlib.Path,required=True);ap.add_argument('--output',type=pathlib.Path,required=True);a=ap.parse_args()
    result=analyze(a.input);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['by_mode'],indent=2))
