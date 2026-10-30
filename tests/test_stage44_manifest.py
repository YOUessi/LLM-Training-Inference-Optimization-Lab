import hashlib,json,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class IFBenchExternalManifestTests(unittest.TestCase):
 def test_60_fixed_multifamily_and_external_source_license(self):
  m=json.loads((ROOT/'data/stage44_ifbench_manifest.json').read_text())
  self.assertEqual(m['sample_count'],60)
  self.assertEqual(sum(m['quota_by_family'].values()),60)
  self.assertEqual(len(m['source_keys_and_prompt_hashes']),60)
  self.assertEqual(len({x['key'] for x in m['source_keys_and_prompt_hashes']}),60)
  self.assertEqual(m['source_original_sha256'],'a2bf7b8e7ed11d39c65b4ec781da8b99663ed0121f97033d70fcecb61f8a9d85')
  self.assertEqual(m['source_commit'],'1c40f0c10d9b5c5c2f10a175a28007ebb64f7f4d')
  self.assertTrue(m['private_raw_text_not_committed'])
 def test_unmodified_official_ifbench_code_present(self):
  self.assertTrue((ROOT/'third_party/ifbench/ifbench/instructions.py').is_file())
  self.assertTrue((ROOT/'third_party/ifbench/evaluation_lib.py').is_file())
  self.assertTrue((ROOT/'third_party/ifbench/LICENSE').is_file())
