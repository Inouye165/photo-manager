import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.intelligence_core import (
    ActiveLearningConfig,
    IntelligenceConfig,
    IntelligenceCore,
    VectorStoreConfig,
)


class TestIntelligenceCore(unittest.TestCase):
    def test_vector_upsert_and_yes_no_queue(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            config = IntelligenceConfig(
                workspace_root=workspace,
                vector_store=VectorStoreConfig(provider="in_memory"),
                active_learning=ActiveLearningConfig(
                    auto_accept_threshold=0.995,
                    verify_threshold=0.80,
                    reject_threshold=0.40,
                    schedule_fine_tune=False,
                ),
            )
            core = IntelligenceCore(config)

            core.learn_from_label(
                relative_path="people/ron_001.jpg",
                image_path=None,
                identity_label="Ron",
                subject_type="people",
                embedding=np.array([1.0, 0.0, 0.0], dtype=np.float32),
                schedule_fine_tune=False,
            )

            task = core.propose_identity(
                relative_path="people/ron_candidate.jpg",
                subject_type="people",
                embedding=np.array([0.8, 0.6, 0.0], dtype=np.float32),
            )

            self.assertEqual(task.status, "needs_confirmation")
            self.assertEqual(task.proposed_label, "Ron")
            self.assertEqual(task.next_action, "queue_yes_no")

    def test_new_identity_when_no_hits_exist(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            core = IntelligenceCore(
                IntelligenceConfig(
                    workspace_root=workspace,
                    vector_store=VectorStoreConfig(provider="in_memory"),
                    active_learning=ActiveLearningConfig(schedule_fine_tune=False),
                )
            )

            task = core.propose_identity(
                relative_path="animals/cat_candidate.jpg",
                subject_type="animals",
                embedding=np.array([0.0, 1.0, 0.0], dtype=np.float32),
            )

            self.assertEqual(task.status, "new_identity")
            self.assertEqual(task.next_action, "collect_label")