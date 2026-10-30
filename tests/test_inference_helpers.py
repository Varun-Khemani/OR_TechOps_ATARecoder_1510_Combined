import unittest

import numpy as np

from ata_recoder_seed42.mongo_shadow import mongo_uri, prediction_document
from ata_recoder_seed42.ranking import joint_topk, normalize_model_text


class RankingTests(unittest.TestCase):
    def test_normalize_model_text_only_collapses_whitespace(self):
        self.assertEqual(normalize_model_text("  HYD   SYS\nFAULT "), "HYD SYS FAULT")
        self.assertEqual(normalize_model_text(None), "")

    def test_joint_topk_ranks_with_chapter_prior_and_returns_code_probability(self):
        probs2 = np.array([0.9, 0.1])
        probs4 = np.array([0.3, 0.7])
        parent_ids = np.array([0, 1])
        top_ids, scores = joint_topk(probs2, probs4, parent_ids, k=2, alpha=1.0)
        self.assertEqual(top_ids.tolist(), [0, 1])
        self.assertTrue(np.allclose(scores, [0.3, 0.7]))

    def test_password_secret_builds_escaped_legacy_connection_string(self):
        from unittest.mock import patch

        with patch.dict(
            "os.environ",
            {
                "MONGODB_URI": "",
                "MONGODB_USER": "app user",
                "MONGODB_HOST": "mongo.example",
            },
        ):
            uri = mongo_uri_from_value("p@ss word")
        self.assertIn("app+user:p%40ss+word@mongo.example/", uri)

    def test_full_mongodb_uri_is_returned_unchanged(self):
        from unittest.mock import patch

        uri = "mongodb+srv://example.invalid/db"
        with patch.dict("os.environ", {"MONGODB_URI": uri}):
            self.assertEqual(mongo_uri(None), uri)

    def test_shadow_document_never_enables_auto_recode(self):
        from datetime import datetime, timezone

        predictions = [type("P", (), {"code": "0052", "score": 0.7})() for _ in range(5)]
        document = prediction_document(predictions, datetime.now(timezone.utc))
        self.assertFalse(document["autoRecode"])
        self.assertTrue(document["qaReview"])
        self.assertEqual(document["ataPred1"], "0052")


def mongo_uri_from_value(value):
    from tempfile import NamedTemporaryFile

    with NamedTemporaryFile("w", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        return mongo_uri(handle.name)


if __name__ == "__main__":
    unittest.main()
