"""Unlimited-admission baseline for Stage22 frozen stdlib inputs and seeds."""
from llm_opt_lab.serving import stage14_mixed_bench as bench
from llm_opt_lab.serving.stage22_stdlib_bench import STDLIB_DOMAINS
from llm_opt_lab.serving.stage13_policy import AdmissionGate

def main(argv=None):
    old_domains,old_policies,old_gate=bench.DOMAINS,bench.POLICIES,bench.AblationAdmissionGate
    try:
        bench.DOMAINS=STDLIB_DOMAINS
        bench.POLICIES=('unlimited',)
        bench.AblationAdmissionGate=lambda policy:AdmissionGate(policy)
        return bench.main(argv)
    finally:
        bench.DOMAINS,bench.POLICIES,bench.AblationAdmissionGate=old_domains,old_policies,old_gate

if __name__=='__main__':raise SystemExit(main())
