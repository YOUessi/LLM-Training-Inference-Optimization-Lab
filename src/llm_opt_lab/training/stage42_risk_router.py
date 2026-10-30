"""A stateless, deterministic response-constraint-sensitive base/LoRA router.

No IFEval grader, benchmark id, category or expected answer is consulted by
choose_model(). Exact same prompt-only rule used for real IFEval and Dolly.
"""
from __future__ import annotations
import re
from dataclasses import dataclass

# Risk scores are fixed engineering priors on explicit *user-request constraints*,
# not fit or calibrated on the frozen test results.
PATTERNS=(
 ('exact_amount',r'\b(?:exactly|at least|at most|no more than|more than|fewer than|less than)\b',2),
 ('format',r'\b(?:json|xml|csv|yaml|markdown|bullet(?: points?)?|code block|uppercase|lowercase|capital letters|bold|italic)\b',3),
 ('literal_start_end',r'\b(?:start (?:with|by)|begin with|end with|finish with|first word|last word)\b',3),
 ('counting',r'\b(?:words?|sentences?|paragraphs?|characters?|letters?|lines?|punctuation|commas?|hashtags?|symbols?|emojis?)\b',2),
 ('exclusion',r'\b(?:do not|don.t|never|without|avoid|exclude|must not|shall not|cannot)\b',2),
 ('explicit_compliance',r'\b(?:must|only|strictly|your (?:answer|response|output)|response (?:must|should)|should contain)\b',1),
 ('hard_keywords',r'\b(?:include|contain|mention|repeat|write the word|use the exact|use only)\b',1),
 ('language_constraint',r'\b(?:respond|answer|reply|write|output) (?:only )?in (?:english|french|spanish|german|chinese|japanese|russian|arabic|hindi)\b',3),
 ('structured_layout',r'\b(?:numbered list|table with|separate sections|list of|heading[s]?|title[s]?)\b',2),
)
THRESHOLD=2

@dataclass(frozen=True)
class RouteDecision:
 choice: str
 risk_score: int
 features: tuple[str,...]


def choose_model(prompt:str)->RouteDecision:
 """Route hard-constrained requests to base, other prompts to Dolly-trained LoRA.

 No external data, benchmark-specific signal or output scoring is permitted.
 """
 if not isinstance(prompt,str) or not prompt.strip():raise ValueError('non-empty user prompt required')
 if len(prompt)>25000:raise ValueError('unbounded prompt not supported')
 found=[];score=0
 for label,pattern,weight in PATTERNS:
  if re.search(pattern,prompt,flags=re.IGNORECASE):
   found.append(label);score+=weight
 return RouteDecision(choice='base' if score>=THRESHOLD else 'lora_sft',
                      risk_score=score,features=tuple(found))
