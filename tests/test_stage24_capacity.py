"""Guard supported Qwen3-AWQ serving slot capacities."""
import subprocess,os,sys,unittest
class CapacityTests(unittest.TestCase):
 def test_driver_capacity_control_env(self):
  for val in ('8','16'):
   env=os.environ.copy();env['PYTHONPATH']='src';env['VLLM_MAX_NUM_SEQS']=val
   p=subprocess.run([sys.executable,'-c',
      'from llm_opt_lab.serving.stage14_mixed_bench import SERVER_MAX_SEQS;print(SERVER_MAX_SEQS)'],
      cwd=os.path.dirname(os.path.dirname(__file__)),env=env,text=True,capture_output=True,check=False)
   self.assertEqual(p.returncode,0,p.stderr)
   self.assertEqual(p.stdout.strip(),val)
 def test_reject_unreviewed_capacity(self):
  env=os.environ.copy();env['PYTHONPATH']='src';env['VLLM_MAX_NUM_SEQS']='64'
  p=subprocess.run([sys.executable,'-c','import llm_opt_lab.serving.stage14_mixed_bench'],
     cwd=os.path.dirname(os.path.dirname(__file__)),env=env,text=True,capture_output=True,check=False)
  self.assertNotEqual(p.returncode,0)
  self.assertIn('ValueError',p.stderr)
