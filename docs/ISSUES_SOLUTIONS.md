# Issues, decisions and resolutions

| Date | Issue | Root cause | Resolution / state | Verification |
|---|---|---|---|---|
| 2026-10-26 | Existing GPU debugger excludes performance profiling | V2 spec focuses on correctness and fault repair; Nsight is an explicit non-goal | Create a distinct systems-optimization lab instead of expanding debugger scope | Read pinned V2 spec and working implementation branch |
| 2026-10-26 | CodeRL-Lab is a post-training study, not a serving engine | Uses TRL/GRPO research pipeline | Reuse benchmark principles, write separate training infrastructure demo | Read pinned CodeRL-Lab README and train module |
| 2026-10-26 | GitHub connected actions cannot create a new repository | Connected GitHub integration supports existing repository writes but lacks create-repo action; Mac SSH auth alone cannot invoke GitHub repository creation API | New Git repo initialized locally; remote GitHub creation/push pending one-time user action | Git CLI and account permissions checked |
| 2026-10-26 | Mac does not have Torch/Triton/NVIDIA GPU tools | Development host is Mac with Python 3.12 and standard library, no torch/triton | CPU smoke on Mac; GPU experiments gated for Tang | Dependency/environment inspection |
| 2026-10-26 | Need to preserve real measurements | No verified GPU benchmark outputs in new repo yet | All performance scripts fail closed if no CUDA; no fabricated speedups | Pending actual run |
| 2026-10-26 | First local CPU validation | Basic stdlib tests and CLI contracts needed | All 24 unittest cases passed, Python compileall and two bash -n checks passed | Mac Python 3.12 console output (not a GPU benchmark) |
