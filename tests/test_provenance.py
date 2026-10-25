from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from llm_opt_lab.provenance import environment, write_json_atomic


class ProvenanceTests(unittest.TestCase):
    def test_environment_safe_shape(self):
        info=environment()
        self.assertIn("python",info)
        self.assertIn("git_commit",info)
        self.assertNotIn("username",info)
        self.assertNotIn("access_token",info)

    def test_atomic_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/"nested"/"metric.json"
            write_json_atomic(output,{"a":3,"status":"not_measured"})
            self.assertEqual(json.loads(output.read_text())["a"],3)
            self.assertEqual(list(output.parent.glob("*.tmp-*")),[])

    def test_nan_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/"result.json"
            with self.assertRaises(ValueError):
                write_json_atomic(output,{"bad":float("nan")})
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
