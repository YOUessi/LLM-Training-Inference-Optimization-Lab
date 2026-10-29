import unittest
from unittest.mock import patch
from llm_opt_lab.serving.stage35_human_bench import prefix_cache_counters
class Response:
 def __init__(self,text):self.value=text.encode()
 def __enter__(self):return self
 def __exit__(self,*args):return None
 def read(self):return self.value
class VllmPrefixCounterTests(unittest.TestCase):
 def test_actual_query_hit_token_counters(self):
  text='# HELP vllm:prefix_cache_hits_total real token hits\n'+(
   'vllm:prefix_cache_queries_total{model_name="local"} 2048\n'+
   'vllm:prefix_cache_hits_total{model_name="local"} 1024\n')
  with patch('urllib.request.urlopen',return_value=Response(text)):
   self.assertEqual(prefix_cache_counters(),{'queries':2048.,'hits':1024.})
 def test_missing_counter_fails_closed(self):
  with patch('urllib.request.urlopen',return_value=Response('vllm:prefix_cache_hits_total 12\n')):
   with self.assertRaises(ValueError):prefix_cache_counters()
