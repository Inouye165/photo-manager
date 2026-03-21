"""Regression tests for the active-learning intelligence core."""

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
    """Verify ranking, class filtering, and negative-memory behaviors."""

    def test_vector_upsert_and_yes_no_queue(self):
        """A near match should land in the confirmation queue instead of auto-accept."""

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
                subject_type="person",
                class_name="person",
                embedding=np.array([1.0, 0.0, 0.0], dtype=np.float32),
                schedule_fine_tune=False,
            )

            task = core.propose_identity(
                relative_path="people/ron_candidate.jpg",
                subject_type="person",
                class_name="person",
                embedding=np.array([0.8, 0.6, 0.0], dtype=np.float32),
            )

            self.assertEqual(task.status, "needs_confirmation")
            self.assertEqual(task.proposed_label, "Ron")
            self.assertEqual(task.next_action, "queue_yes_no")

    def test_new_identity_when_no_hits_exist(self):
        """Queries without any candidates should require a fresh label."""

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
                class_name="cat",
                embedding=np.array([0.0, 1.0, 0.0], dtype=np.float32),
            )

            self.assertEqual(task.status, "new_identity")
            self.assertEqual(task.next_action, "collect_label")

    def test_species_strictness_blocks_cross_species_matches(self):
        """A dog prototype must not be proposed for a bird detection."""

        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            core = IntelligenceCore(
                IntelligenceConfig(
                    workspace_root=workspace,
                    vector_store=VectorStoreConfig(provider="in_memory"),
                    active_learning=ActiveLearningConfig(schedule_fine_tune=False),
                )
            )

            core.learn_from_label(
                relative_path="animals/dog_001.jpg",
                image_path=None,
                identity_label="Dobby",
                subject_type="dog",
                class_name="dog",
                embedding=np.array([1.0, 0.0, 0.0], dtype=np.float32),
                schedule_fine_tune=False,
            )

            task = core.propose_identity(
                relative_path="animals/bird_candidate.jpg",
                subject_type="bird",
                class_name="bird",
                embedding=np.array([1.0, 0.0, 0.0], dtype=np.float32),
            )

            self.assertEqual(task.status, "new_identity")
            self.assertIsNone(task.proposed_label)

    def test_semantic_search_returns_matching_label(self):
        """Text search should find the label stored with the same semantic vector."""

        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            core = IntelligenceCore(
                IntelligenceConfig(
                    workspace_root=workspace,
                    vector_store=VectorStoreConfig(provider="in_memory"),
                    active_learning=ActiveLearningConfig(schedule_fine_tune=False, top_k=3),
                )
            )

            vector = core.embed_text("golden retriever")
            core.learn_from_label(
                relative_path="animals/dog_001.jpg",
                image_path=None,
                identity_label="Dobby",
                subject_type="dog",
                class_name="dog",
                embedding=vector,
                schedule_fine_tune=False,
            )

            hits = core.semantic_search(query="golden retriever", top_k=2)
            self.assertEqual(len(hits), 1)
            self.assertEqual(hits[0].identity_label, "Dobby")

    def test_record_id_overwrites_existing_positive_embedding(self):
        """Reusing a stable record id should replace the prior positive embedding."""

        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            query_vector = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            core = IntelligenceCore(
                IntelligenceConfig(
                    workspace_root=workspace,
                    vector_store=VectorStoreConfig(provider="in_memory"),
                    active_learning=ActiveLearningConfig(schedule_fine_tune=False, top_k=3),
                )
            )

            core.learn_from_label(
                record_id="working_dir/img1.jpg#0",
                relative_path="working_dir/img1.jpg#0",
                image_path=None,
                identity_label="Ron",
                subject_type="person",
                class_name="person",
                embedding=query_vector,
                schedule_fine_tune=False,
            )
            core.learn_from_label(
                record_id="working_dir/img1.jpg#0",
                relative_path="working_dir/img1.jpg#0",
                image_path=None,
                identity_label="Trisha",
                subject_type="person",
                class_name="person",
                embedding=query_vector,
                schedule_fine_tune=False,
            )

            hits = core.vector_index.query(
                query_vector,
                subject_type="person",
                top_k=5,
                class_name="person",
            )
            self.assertEqual(len(hits), 1)
            self.assertEqual(hits[0].identity_label, "Trisha")

    def test_negative_samples_penalize_bad_matches(self):
        """Rejected labels should be penalized so a weaker good match can win."""

        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            core = IntelligenceCore(
                IntelligenceConfig(
                    workspace_root=workspace,
                    vector_store=VectorStoreConfig(provider="in_memory"),
                    active_learning=ActiveLearningConfig(schedule_fine_tune=False, top_k=3),
                )
            )

            query_vector = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            core.learn_from_label(
                record_id="dog-1",
                relative_path="dog-1",
                image_path=None,
                identity_label="Dobby",
                subject_type="dog",
                class_name="dog",
                embedding=query_vector,
                schedule_fine_tune=False,
            )
            core.learn_from_label(
                record_id="dog-2",
                relative_path="dog-2",
                image_path=None,
                identity_label="Sawyer",
                subject_type="dog",
                class_name="dog",
                embedding=np.array([0.8, 0.2, 0.0], dtype=np.float32),
                schedule_fine_tune=False,
            )
            core.add_negative_sample(
                record_id="negative::dog-1::dobby",
                relative_path="dog-1",
                negative_label="Dobby",
                image_path=None,
                subject_type="dog",
                class_name="dog",
                embedding=query_vector,
            )

            task = core.propose_identity(
                relative_path="candidate",
                subject_type="dog",
                class_name="dog",
                embedding=query_vector,
            )

            self.assertEqual(task.proposed_label, "Sawyer")
            self.assertGreater(task.hits[0].score, task.hits[1].score)
