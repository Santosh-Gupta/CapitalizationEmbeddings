import unittest
from pathlib import Path
from unittest.mock import patch

from capitalization_embeddings import runtime_paths


class RuntimePathTests(unittest.TestCase):
    def test_workspace_prefers_runpod_workspace(self):
        def exists(path):
            return str(path) == "/workspace"

        with patch.object(Path, "exists", exists):
            self.assertEqual(runtime_paths.workspace_root(), Path("/workspace"))

    def test_workspace_uses_colab_drive_without_runpod(self):
        def exists(path):
            return str(path) == "/content/drive/MyDrive"

        with patch.object(Path, "exists", exists):
            self.assertEqual(runtime_paths.workspace_root(), Path("/content/drive/MyDrive"))

    def test_workspace_falls_back_to_repo_root(self):
        with patch.object(Path, "exists", lambda path: False):
            self.assertEqual(runtime_paths.workspace_root(), runtime_paths.repo_root())

    def test_configure_huggingface_cache_sets_env(self):
        with patch.dict("os.environ", {}, clear=True):
            cache = runtime_paths.configure_huggingface_cache()

            self.assertTrue(cache.endswith(".cache/huggingface"))
            self.assertEqual(runtime_paths.os.environ["HF_HOME"], cache)
            self.assertIn("HF_DATASETS_CACHE", runtime_paths.os.environ)


if __name__ == "__main__":
    unittest.main()
