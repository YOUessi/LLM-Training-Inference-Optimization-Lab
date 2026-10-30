"""Disjoint rule-checking dev/final test and fixed guardrail configuration."""
import hashlib,json,unittest
from pathlib import Path
from llm_opt_lab.training.stage38_adapter_strength import load_split,SCALES,DEV_MAX_LOSS_IN_STRICT,DEV_MIN_BASE_CE_IMPROVEMENT,DEV_MAX_EXTRA_CAP_HITS
ROOT=Path(__file__).resolve().parents[1]
class Stage38HoldoutAndSafetyTests(unittest.TestCase):
 def test_ifeval_exclude_known96_dev43_test101(self):
  dev,manifest=load_split(ROOT/'data','dev')
  test,_=load_split(ROOT/'data','test')
  existing=[json.loads(x) for x in (ROOT/'data/stage36_ifeval_96.jsonl').read_text().splitlines()]
  sets=[{r['key'] for r in rows} for rows in (existing,dev,test)]
  self.assertEqual([len(x) for x in sets],[96,43,101])
  self.assertFalse(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
  self.assertTrue(manifest['test_must_remain_untouched_until_scale_selected'])
  self.assertEqual(set(manifest['class_assignments']),{'change_case','combination','detectable_content','detectable_format','keywords','language','length_constraints','punctuation','startend'})
 def test_predeclared_scales_and_both_goals(self):
  self.assertEqual(SCALES,(0.,.25,.5,.75,1.))
  self.assertEqual(DEV_MAX_LOSS_IN_STRICT,2)
  self.assertEqual(DEV_MAX_EXTRA_CAP_HITS,2)
  self.assertAlmostEqual(DEV_MIN_BASE_CE_IMPROVEMENT,.1)
