import hashlib,json,unittest
from pathlib import Path
from llm_opt_lab.training.stage36_ifeval import sources,MODELS
ROOT=Path(__file__).resolve().parents[1]
class IFEvalPreRegistrationTests(unittest.TestCase):
 def test_96_stratified_verified_original_data(self):
  rows,manifest=sources(ROOT/'data')
  self.assertEqual(len(rows),96)
  self.assertEqual(len({x['key'] for x in rows}),96)
  self.assertEqual(manifest['source_original_sha256'],'6a85310ca8ce15eff755aa08a3a4ff931c7e273e7515ebb3c492ea85fd8288f2')
  self.assertEqual(set(manifest['category_counts']),{'punctuation','detectable_content','combination','change_case','detectable_format','length_constraints','language','startend','keywords'})
  self.assertEqual(set(MODELS),{'base','stage26_short','stage31_full'})
  self.assertEqual(manifest['max_generated_tokens'],384)
 def test_official_google_rule_based_evaluator(self):
  from instruction_following_eval import evaluation_lib
  ex=evaluation_lib.InputExample(key=1001,instruction_id_list=['punctuation:no_comma'],
   prompt='Please do not write commas.',kwargs=[{}])
  correct=evaluation_lib.test_instruction_following_strict(ex,{ex.prompt:'This is a correct response without commas.'})
  wrong=evaluation_lib.test_instruction_following_strict(ex,{ex.prompt:'This, has a comma.'})
  self.assertTrue(correct.follow_all_instructions)
  self.assertFalse(wrong.follow_all_instructions)
if __name__=='__main__':unittest.main()
