"""
Unit tests for Module 4 Stage 4: Backward Chaining.
Verifies all 22 required backward-chaining capabilities:
1. Goal already present in WorkingMemory.
2. Goal proved using R1.
3. Goal proved using R2.
4. Goal proved using R3.
5. Goal proved using R4.
6. Multi-step recursive proof.
7. Multiple possible proof paths.
8. Missing premise.
9. Unsupported goal.
10. Repeated goal detection.
11. Cyclic dependency detection.
12. Maximum recursion depth.
13. Maximum rule evaluation limit.
14. Correct supporting fact IDs.
15. Correct rule IDs in proof traces.
16. Intermediate subgoal tracking.
17. Original facts remain unchanged.
18. Inferred facts are not inserted into WorkingMemory.
19. WorkingMemory remains unchanged after proof attempts.
20. Empty WorkingMemory.
21. Case-insensitive goal matching.
22. Full test suite regression.
"""

import unittest

from modules.reasoning_engine.reasoning_models import Fact, InferenceRule
from modules.reasoning_engine.working_memory import WorkingMemory
from modules.reasoning_engine.rule_base import (
    R1,
    R2,
    R3,
    R4,
    get_all_rules,
)
from modules.reasoning_engine.backward_chaining import (
    BackwardChainer,
    backward_chain,
    ProofResult,
    ProofStatus,
)


class TestBackwardChaining(unittest.TestCase):
    """Exhaustive test suite for Backward Chaining goal-driven deduction."""

    def setUp(self):
        self.wm = WorkingMemory()
        self.chainer = BackwardChainer()

    # 1. Goal already present in WorkingMemory
    def test_01_goal_already_present_in_working_memory(self):
        self.wm.add_original_fact("Paris", "capital of", "France", fact_id=10)

        result = self.chainer.prove(("Paris", "capital of", "France"), self.wm)

        self.assertTrue(result.is_proven)
        self.assertTrue(result.directly_found)
        self.assertEqual(result.status, "proven")
        self.assertEqual(result.supporting_fact_ids, [10])
        self.assertEqual(len(result.applied_rule_ids), 0)

    # 2. Goal proved using R1
    def test_02_goal_proved_using_r1(self):
        self.wm.add_original_fact("Bangalore", "capital of", "Karnataka", fact_id=1)
        self.wm.add_original_fact("Karnataka", "is a state in", "India", fact_id=2)

        goal = ("Bangalore", "located in", "India")
        result = self.chainer.prove(goal, self.wm, rules=[R1])

        self.assertTrue(result.is_proven)
        self.assertFalse(result.directly_found)
        self.assertIn("R1", result.applied_rule_ids)
        self.assertEqual(set(result.supporting_fact_ids), {1, 2})

    # 3. Goal proved using R2
    def test_03_goal_proved_using_r2(self):
        self.wm.add_original_fact("Manhattan", "located in", "New York City", fact_id=11)
        self.wm.add_original_fact("New York City", "located in", "USA", fact_id=12)

        goal = ("Manhattan", "located in", "USA")
        result = self.chainer.prove(goal, self.wm, rules=[R2])

        self.assertTrue(result.is_proven)
        self.assertFalse(result.directly_found)
        self.assertIn("R2", result.applied_rule_ids)
        self.assertEqual(set(result.supporting_fact_ids), {11, 12})

    # 4. Goal proved using R3
    def test_04_goal_proved_using_r3(self):
        self.wm.add_original_fact("Piston", "part of", "Engine", fact_id=21)
        self.wm.add_original_fact("Engine", "part of", "Car", fact_id=22)

        goal = ("Piston", "part of", "Car")
        result = self.chainer.prove(goal, self.wm, rules=[R3])

        self.assertTrue(result.is_proven)
        self.assertIn("R3", result.applied_rule_ids)
        self.assertEqual(set(result.supporting_fact_ids), {21, 22})

    # 5. Goal proved using R4
    def test_05_goal_proved_using_r4(self):
        self.wm.add_original_fact("Canada", "borders", "USA", fact_id=31)

        goal = ("USA", "borders", "Canada")
        result = self.chainer.prove(goal, self.wm, rules=[R4])

        self.assertTrue(result.is_proven)
        self.assertIn("R4", result.applied_rule_ids)
        self.assertEqual(result.supporting_fact_ids, [31])

    # 6. Multi-step recursive proof
    def test_06_multi_step_recursive_proof(self):
        # 3-hop transitivity: Bangalore -> Karnataka -> India -> Asia
        self.wm.add_original_fact("Bangalore", "located in", "Karnataka", fact_id=1)
        self.wm.add_original_fact("Karnataka", "located in", "India", fact_id=2)
        self.wm.add_original_fact("India", "located in", "Asia", fact_id=3)

        goal = ("Bangalore", "located in", "Asia")
        result = self.chainer.prove(goal, self.wm, rules=[R2])

        self.assertTrue(result.is_proven)
        self.assertEqual(set(result.supporting_fact_ids), {1, 2, 3})
        # R2 fired multiple times along the proof chain
        self.assertTrue(len(result.applied_rule_ids) >= 2)

    # 7. Multiple possible proof paths
    def test_07_multiple_possible_proof_paths(self):
        # Path 1: A -> B (dead end, no B -> D)
        self.wm.add_original_fact("A", "located in", "B", fact_id=1)
        # Path 2: A -> C -> D (successful path)
        self.wm.add_original_fact("A", "located in", "C", fact_id=2)
        self.wm.add_original_fact("C", "located in", "D", fact_id=3)

        goal = ("A", "located in", "D")
        result = self.chainer.prove(goal, self.wm, rules=[R2])

        self.assertTrue(result.is_proven)
        self.assertEqual(set(result.supporting_fact_ids), {2, 3})

    # 8. Missing premise
    def test_08_missing_premise(self):
        # Premise 1 is present, but Premise 2 is missing
        self.wm.add_original_fact("Bangalore", "capital of", "Karnataka", fact_id=1)

        goal = ("Bangalore", "located in", "India")
        result = self.chainer.prove(goal, self.wm, rules=[R1])

        self.assertFalse(result.is_proven)
        self.assertEqual(result.status, "unproven")
        self.assertEqual(len(result.supporting_fact_ids), 0)

    # 9. Unsupported goal
    def test_09_unsupported_goal(self):
        self.wm.add_original_fact("Python", "is a", "programming language", fact_id=1)

        goal = ("Python", "invented by", "Guido van Rossum")
        result = self.chainer.prove(goal, self.wm)

        self.assertFalse(result.is_proven)
        self.assertEqual(result.status, "unproven")

    # 10. Repeated goal detection
    def test_10_repeated_goal_detection(self):
        self.wm.add_original_fact("A", "borders", "B", fact_id=1)

        # Proving the same goal consecutively
        res1 = self.chainer.prove(("B", "borders", "A"), self.wm, rules=[R4])
        res2 = self.chainer.prove(("B", "borders", "A"), self.wm, rules=[R4])

        self.assertTrue(res1.is_proven)
        self.assertTrue(res2.is_proven)
        self.assertEqual(res1.supporting_fact_ids, res2.supporting_fact_ids)

    # 11. Cyclic dependency detection
    def test_11_cyclic_dependency_detection(self):
        # Mutual border loop: if A borders B was goal, and R4 asks for B borders A
        # but neither fact exists in memory
        result = self.chainer.prove(("France", "borders", "Germany"), self.wm, rules=[R4])

        # Must terminate safely without infinite recursion
        self.assertFalse(result.is_proven)
        self.assertIn(result.status, ("unproven", "cycle_detected"))

    # 12. Maximum recursion depth
    def test_12_maximum_recursion_depth(self):
        # Chain requiring depth >= 2
        self.wm.add_original_fact("A", "located in", "B", fact_id=1)
        self.wm.add_original_fact("B", "located in", "C", fact_id=2)
        self.wm.add_original_fact("C", "located in", "D", fact_id=3)

        # Force max_depth=1
        limited_chainer = BackwardChainer(rules=[R2], max_depth=1)
        result = limited_chainer.prove(("A", "located in", "D"), self.wm)

        self.assertFalse(result.is_proven)
        self.assertEqual(result.status, "max_depth_reached")

    # 13. Maximum rule evaluation limit
    def test_13_maximum_rule_evaluation_limit(self):
        self.wm.add_original_fact("A", "located in", "B", fact_id=1)
        self.wm.add_original_fact("B", "located in", "C", fact_id=2)

        # Force max_evaluations=1
        limited_chainer = BackwardChainer(rules=[R2], max_evaluations=1)
        result = limited_chainer.prove(("A", "located in", "C"), self.wm)

        # If evaluations limit is 1, proving intermediate subgoals exceeds limit
        self.assertIn(result.status, ("max_evaluations_reached", "unproven"))

    # 14. Correct supporting fact IDs
    def test_14_correct_supporting_fact_ids(self):
        self.wm.add_original_fact("Bangalore", "capital of", "Karnataka", fact_id=501)
        self.wm.add_original_fact("Karnataka", "is a state in", "India", fact_id=502)

        result = self.chainer.prove(("Bangalore", "located in", "India"), self.wm, rules=[R1])
        self.assertEqual(result.supporting_fact_ids, [501, 502])

    # 15. Correct rule IDs in proof traces
    def test_15_correct_rule_ids_in_proof_traces(self):
        self.wm.add_original_fact("Bangalore", "capital of", "Karnataka", fact_id=1)
        self.wm.add_original_fact("Karnataka", "is a state in", "India", fact_id=2)

        result = self.chainer.prove(("Bangalore", "located in", "India"), self.wm, rules=[R1])
        self.assertEqual(result.applied_rule_ids, ["R1"])
        traces = result.inference_traces
        self.assertTrue(any(t.applied_rule_id == "R1" for t in traces))

    # 16. Intermediate subgoal tracking
    def test_16_intermediate_subgoal_tracking(self):
        self.wm.add_original_fact("Bangalore", "capital of", "Karnataka", fact_id=1)
        self.wm.add_original_fact("Karnataka", "is a state in", "India", fact_id=2)

        result = self.chainer.prove(("Bangalore", "located in", "India"), self.wm, rules=[R1])
        self.assertTrue(len(result.subgoals) > 0)
        # Check that Premise 2 was explored as an intermediate subgoal
        subgoals_normalized = [(s.lower(), p.lower(), o.lower()) for s, p, o in result.subgoals]
        self.assertIn(("karnataka", "is a state in", "india"), subgoals_normalized)

    # 17. Original facts remain unchanged
    def test_17_original_facts_remain_unchanged(self):
        self.wm.add_original_fact("Bangalore", "capital of", "Karnataka", fact_id=1)
        self.wm.add_original_fact("Karnataka", "is a state in", "India", fact_id=2)

        count_before = self.wm.count_original()
        self.chainer.prove(("Bangalore", "located in", "India"), self.wm, rules=[R1])
        count_after = self.wm.count_original()

        self.assertEqual(count_before, count_after)

    # 18. Inferred facts are not inserted into WorkingMemory
    def test_18_inferred_facts_not_inserted_into_working_memory(self):
        self.wm.add_original_fact("Bangalore", "capital of", "Karnataka", fact_id=1)
        self.wm.add_original_fact("Karnataka", "is a state in", "India", fact_id=2)

        self.chainer.prove(("Bangalore", "located in", "India"), self.wm, rules=[R1])

        # WorkingMemory should have 0 inferred facts
        self.assertEqual(self.wm.count_inferred(), 0)
        self.assertFalse(self.wm.contains_triple("Bangalore", "located in", "India"))

    # 19. WorkingMemory remains unchanged after proof attempts
    def test_19_working_memory_remains_unchanged_after_proof_attempts(self):
        self.wm.add_original_fact("A", "borders", "B", fact_id=1)

        total_before = self.wm.count_total()
        self.chainer.prove(("B", "borders", "A"), self.wm, rules=[R4])
        self.chainer.prove(("NonExistent", "borders", "Place"), self.wm)
        total_after = self.wm.count_total()

        self.assertEqual(total_before, total_after)

    # 20. Empty WorkingMemory
    def test_20_empty_working_memory(self):
        result = self.chainer.prove(("Anything", "located in", "Somewhere"), self.wm)
        self.assertFalse(result.is_proven)
        self.assertEqual(result.status, "unproven")

    # 21. Case-insensitive goal matching
    def test_21_case_insensitive_goal_matching(self):
        self.wm.add_original_fact("Bangalore", "capital of", "Karnataka", fact_id=1)
        self.wm.add_original_fact("Karnataka", "is a state in", "India", fact_id=2)

        # Query with lowercase / uppercase casing
        result = self.chainer.prove(("BANGALORE", "LOCATED IN", "india"), self.wm, rules=[R1])
        self.assertTrue(result.is_proven)

    # 22. Full test suite regression
    def test_22_full_test_suite_regression(self):
        # Comprehensive test checking backward_chain convenience function
        self.wm.add_original_fact("India", "borders", "Nepal", fact_id=9)
        result = backward_chain(("Nepal", "borders", "India"), self.wm, rules=get_all_rules())
        self.assertTrue(result.is_proven)
        self.assertEqual(result.supporting_fact_ids, [9])
        self.assertIn("R4", result.applied_rule_ids)


if __name__ == '__main__':
    unittest.main()
