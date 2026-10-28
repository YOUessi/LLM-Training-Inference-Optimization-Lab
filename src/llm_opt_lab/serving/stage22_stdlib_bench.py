"""Independent from Stage15 document domains: PSF-licensed local stdlib docstrings."""
from pathlib import Path
from llm_opt_lab.serving import stage14_mixed_bench as bench

bench.DOMAINS={'stdlib':(Path('data/stage22_stdlib_requests.jsonl'),
    Path('data/stage22_stdlib_manifest.json'),(9301,9309))}

def main(argv=None):
 return bench.main(argv)
if __name__=='__main__':raise SystemExit(main())
