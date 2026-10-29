# Vendored unmodified Google IFEval evaluator (Apache-2.0)

Original repository: https://github.com/google-research/google-research/tree/master/instruction_following_eval

Pinned upstream Git commit: `7e672ccbb111603f237a3daed2f98dccd9300023`.

Source files are copied without modifications: `evaluation_lib.py`, `instructions.py`, `instructions_registry.py`, `instructions_util.py`, requirements, original README. Root Apache-2.0 LICENSE included. Added only local package init files to permit import. Dependencies installed only in Tang isolated Python venv; NLTK `punkt_tab` obtained from official nltk_data Git blob `5e5ff6137d5ee6025e400d1c3a7b21914c48b635` and stored in private non-world-writable `/home/you/nltk_data`, not committed. Evaluator is an official verifiable-instruction **constraint checker**, not a semantic correctness judge.
