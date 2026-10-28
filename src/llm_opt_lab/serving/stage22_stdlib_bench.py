"""Independent PSF-licensed stdlib corpus; preserve Stage14 module globals."""
from pathlib import Path
from llm_opt_lab.serving import stage14_mixed_bench as bench

STDLIB_DOMAINS={'stdlib':(Path('data/stage22_stdlib_requests.jsonl'),
    Path('data/stage22_stdlib_manifest.json'),(9301,9309))}

def main(argv=None):
    # Reuse the driver without mutating global parse choices at import time.
    original=bench.DOMAINS
    try:
        bench.DOMAINS=STDLIB_DOMAINS
        return bench.main(argv)
    finally:
        bench.DOMAINS=original

if __name__=='__main__':raise SystemExit(main())
