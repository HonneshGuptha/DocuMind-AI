"""
Unit tests for Module 4 Stage 3: Forward Chaining.
Verifies all 20 required forward-chaining capabilities:
1. R1 deriving Bangalore located in India.
2. R2 deriving transitive location.
3. R3 deriving transitive part-whole relationships.
4. R4 deriving reverse border relationships.
5. Multiple rules firing in the same run.
6. Multi-step inference requiring repeated iterations.
7. Inferred facts being reused as premises.
8. Correct provenance and supporting fact IDs.
9. InferenceTrace generation.
10. Duplicate conclusions not being inserted twice.
11. No duplicate traces for existing conclusions.
12. Self-relations being rejected.
13. Cycles terminating safely.
14. Empty WorkingMemory.
15. No matching rules.
16. Maximum iteration limit.
17. Maximum inferred-fact limit.
18. Original facts remaining unchanged.
19. Inferred facts remaining separate from original facts.
20. Full inference process terminating at a fixed point.
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
from modules.reasoning_engine.forward_chaining import (
    ForwardChainer,
    forward_chain,
)


class TestForwardChaining(unittest.TestCase):
    """Exhaustive test suite for Forward Chaining data-driven deduction."""

    def setUp(self):
        self.wm = WorkingMemory()
        self.chainer = ForwardChainer()

    # 1. R1 deriving Bangalore located in India
    def test_01_r1_deriving_bangalore_located_in_india(self):
        self.wm.add_original_fact("Bangalore", "capital of", "Karnataka", fact_id=1)
        self.wm.add_original_fact("Karnataka", "is a state in", "India", fact_id=2)

        result = self.chainer.run(self.wm, rules=[R1])

        self.assertEqual(result.status, "success")
        self.assertEqual(len(result.inferred_facts), 1)
        inf = result.inferred_facts[0]
        self.assertEqual(inf.triple, ("Bangalore", "located in", "India"))
        self.assertTrue(inf.is_inferred)
        self.assertEqual(inf.fact_id, "inf_1")

    # 2. R2 deriving transitive location
    def test_02_r2_deriving_transitive_location(self):
        self.wm.add_original_fact("Manhattan", "located in", "New York City", fact_id=10)
        self.wm.add_original_fact("New York City", "located in", "USA", fact_id=11)

        result = self.chainer.run(self.wm, rules=[R2])

        self.assertEqual(result.status, "success")
        self.assertEqual(len(result.inferred_facts), 1)
        self.assertEqual(result.inferred_facts[0].triple, ("Manhattan", "located in", "USA"))

    # 3. R3 deriving transitive part-whole relationships
    def test_03_r3_deriving_transitive_part_whole(self):
        self.wm.add_original_fact("Piston", "part of", "Engine", fact_id=20)
        self.wm.add_original_fact("Engine", "part of", "Car", fact_id=21)

        result = self.chainer.run(self.wm, rules=[R3])

        self.assertEqual(result.status, "success")
        self.assertEqual(len(result.inferred_facts), 1)
        self.assertEqual(result.inferred_facts[0].triple, ("Piston", "part of", "Car"))

    # 4. R4 deriving reverse border relationships
    def test_04_r4_deriving_reverse_border(self):
        self.wm.add_original_fact("France", "borders", "Germany", fact_id=30)

        result = self.chainer.run(self.wm, rules=[R4])

        self.assertEqual(result.status, "success")
        self.assertEqual(len(result.inferred_facts), 1)
        self.assertEqual(result.inferred_facts[0].triple, ("Germany", "borders", "France"))

    # 5. Multiple rules firing in the same run
    def test_05_multiple_rules_firing_in_same_run(self):
        # Facts for R1
        self.wm.add_original_fact("Tokyo", "capital of", "Kanto", fact_id=1)
        self.wm.add_original_fact("Kanto", "is a state in", "Japan", fact_id=2)
        # Facts for R4
        self.wm.add_original_fact("Canada", "borders", "USA", fact_id=3)

        result = self.chainer.run(self.wm, rules=[R1, R4])

        self.assertEqual(result.status, "success")
        self.assertEqual(len(result.inferred_facts), 2)
        triples = [f.triple for f in result.inferred_facts]
        self.assertIn(("Tokyo", "located in", "Japan"), triples)
        self.assertIn(("USA", "borders", "Canada"), triples)

    # 6. Multi-step inference requiring repeated iterations
    def test_06_multi_step_inference_requiring_repeated_iterations(self):
        # 3-hop location chain
        self.wm.add_original_fact("Bangalore", "located in", "Karnataka", fact_id=1)
        self.wm.add_original_fact("Karnataka", "located in", "India", fact_id=2)
        self.wm.add_original_fact("India", "located in", "Asia", fact_id=3)

        result = self.chainer.run(self.wm, rules=[R2])

        self.assertEqual(result.status, "success")
        self.assertEqual(len(result.inferred_facts), 3)
        triples = [f.triple for f in result.inferred_facts]
        self.assertIn(("Bangalore", "located in", "India"), triples)
        self.assertIn(("Karnataka", "located in", "Asia"), triples)
        self.assertIn(("Bangalore", "located in", "Asia"), triples)

    # 7. Inferred facts being reused as premises
    def test_07_inferred_facts_reused_as_premises(self):
        self.wm.add_original_fact("Bangalore", "located in", "Karnataka", fact_id=1)
        self.wm.add_original_fact("Karnataka", "located in", "India", fact_id=2)
        self.wm.add_original_fact("India", "located in", "Asia", fact_id=3)

        result = self.chainer.run(self.wm, rules=[R2])

        # Find the trace that derived (Bangalore, located in, Asia)
        asia_trace = next(
            t for t in result.inference_traces
            if t.derived_fact.triple == ("Bangalore", "located in", "Asia")
        )
        # Check that one of its supporting fact IDs was an inferred fact (starts with 'inf_')
        has_inferred_supporter = any(
            str(fid).startswith("inf_") for fid in asia_trace.supporting_fact_ids
        )
        self.assertTrue(has_inferred_supporter)

    # 8. Correct provenance and supporting fact IDs
    def test_08_correct_provenance_and_supporting_fact_ids(self):
        self.wm.add_original_fact("Bangalore", "capital of", "Karnataka", fact_id=101)
        self.wm.add_original_fact("Karnataka", "is a state in", "India", fact_id=102)

        result = self.chainer.run(self.wm, rules=[R1])

        self.assertEqual(len(result.inference_traces), 1)
        trace = result.inference_traces[0]
        self.assertEqual(trace.applied_rule_id, "R1")
        self.assertEqual(set(trace.supporting_fact_ids), {101, 102})
        self.assertEqual(trace.reasoning_method, "forward_chaining")

    # 9. InferenceTrace generation
    def test_09_inference_trace_generation(self):
        self.wm.add_original_fact("X", "borders", "Y", fact_id=5)
        result = self.chainer.run(self.wm, rules=[R4])

        self.assertEqual(len(result.inference_traces), 1)
        t = result.inference_traces[0]
        self.assertEqual(t.applied_rule_id, "R4")
        self.assertIn("Derived", t.explanation)
        self.assertEqual(t.supporting_fact_ids, [5])

    # 10. Duplicate conclusions not being inserted twice
    def test_10_duplicate_conclusions_not_inserted_twice(self):
        # Fact already exists in memory
        self.wm.add_original_fact("Bangalore", "capital of", "Karnataka", fact_id=1)
        self.wm.add_original_fact("Karnataka", "is a state in", "India", fact_id=2)
        # Conclusion already known as original fact
        self.wm.add_original_fact("Bangalore", "located in", "India", fact_id=3)

        result = self.chainer.run(self.wm, rules=[R1])

        self.assertEqual(result.status, "no_inference")
        self.assertEqual(len(result.inferred_facts), 0)
        self.assertEqual(self.wm.count_total(), 3)

    # 11. No duplicate traces for existing conclusions
    def test_11_no_duplicate_traces_for_existing_conclusions(self):
        self.wm.add_original_fact("A", "borders", "B", fact_id=1)
        self.wm.add_original_fact("B", "borders", "A", fact_id=2)

        result = self.chainer.run(self.wm, rules=[R4])

        self.assertEqual(result.status, "no_inference")
        self.assertEqual(len(result.inference_traces), 0)

    # 12. Self-relations being rejected
    def test_12_self_relations_rejected(self):
        # If border symmetry or transitivity somehow encountered (A, borders, A)
        self.wm.add_original_fact("SelfLand", "borders", "SelfLand", fact_id=1)

        result = self.chainer.run(self.wm, rules=[R4])
        self.assertEqual(len(result.inferred_facts), 0)
        self.assertEqual(result.status, "no_inference")

    # 13. Cycles terminating safely
    def test_13_cycles_terminating_safely(self):
        # Border symmetry cycle: A borders B -> B borders A -> A borders B ...
        self.wm.add_original_fact("Canada", "borders", "USA", fact_id=1)

        result = self.chainer.run(self.wm, rules=[R4])

        # Iteration 1 adds (USA borders Canada)
        # Iteration 2 derives (Canada borders USA), which is already present -> fixed point!
        self.assertEqual(result.status, "success")
        self.assertEqual(len(result.inferred_facts), 1)
        self.assertEqual(self.wm.count_total(), 2)

    # 14. Empty WorkingMemory
    def test_14_empty_working_memory(self):
        result = self.chainer.run(self.wm)

        self.assertEqual(result.status, "no_inference")
        self.assertEqual(len(result.inferred_facts), 0)
        self.assertEqual(len(result.inference_traces), 0)

    # 15. No matching rules
    def test_15_no_matching_rules(self):
        self.wm.add_original_fact("Python", "created by", "Guido van Rossum", fact_id=1)

        result = self.chainer.run(self.wm)

        self.assertEqual(result.status, "no_inference")
        self.assertEqual(len(result.inferred_facts), 0)

    # 16. Maximum iteration limit
    def test_16_maximum_iteration_limit(self):
        # 3-hop chain needs at least 2 derivation iterations to complete
        self.wm.add_original_fact("A", "located in", "B", fact_id=1)
        self.wm.add_original_fact("B", "located in", "C", fact_id=2)
        self.wm.add_original_fact("C", "located in", "D", fact_id=3)

        # Force max_iterations=1
        limited_chainer = ForwardChainer(rules=[R2], max_iterations=1)
        result = limited_chainer.run(self.wm)

        self.assertEqual(result.status, "max_iterations_reached")
        # In iteration 1, (A in C) and (B in D) are derived, but (A in D) requires iteration 2
        self.assertEqual(len(result.inferred_facts), 2)

    # 17. Maximum inferred-fact limit
    def test_17_maximum_inferred_fact_limit(self):
        self.wm.add_original_fact("A", "located in", "B", fact_id=1)
        self.wm.add_original_fact("B", "located in", "C", fact_id=2)
        self.wm.add_original_fact("C", "located in", "D", fact_id=3)

        # Force max_facts=1
        limited_chainer = ForwardChainer(rules=[R2], max_facts=1)
        result = limited_chainer.run(self.wm)

        self.assertEqual(result.status, "max_facts_reached")
        self.assertEqual(len(result.inferred_facts), 1)

    # 18. Original facts remaining unchanged
    def test_18_original_facts_remaining_unchanged(self):
        self.wm.add_original_fact("Bangalore", "capital of", "Karnataka", fact_id=1)
        self.wm.add_original_fact("Karnataka", "is a state in", "India", fact_id=2)

        original_facts_before = self.wm.get_original_facts()

        self.chainer.run(self.wm, rules=[R1])

        original_facts_after = self.wm.get_original_facts()
        self.assertEqual(len(original_facts_before), len(original_facts_after))
        self.assertEqual(original_facts_before[0].triple, original_facts_after[0].triple)
        self.assertEqual(original_facts_before[1].triple, original_facts_after[1].triple)

    # 19. Inferred facts remaining separate from original facts
    def test_19_inferred_facts_remaining_separate_from_original_facts(self):
        self.wm.add_original_fact("Bangalore", "capital of", "Karnataka", fact_id=1)
        self.wm.add_original_fact("Karnataka", "is a state in", "India", fact_id=2)

        result = self.chainer.run(self.wm, rules=[R1])

        self.assertEqual(self.wm.count_original(), 2)
        self.assertEqual(self.wm.count_inferred(), 1)
        self.assertEqual(len(result.original_facts), 2)
        self.assertEqual(len(result.inferred_facts), 1)

        for f in self.wm.get_original_facts():
            self.assertTrue(f.is_original)
            self.assertFalse(f.is_inferred)

        for f in self.wm.get_inferred_facts():
            self.assertTrue(f.is_inferred)
            self.assertFalse(f.is_original)

    # 20. Full inference process terminating at a fixed point
    def test_20_full_inference_process_terminating_at_fixed_point(self):
        # Comprehensive multi-rule domain
        self.wm.add_original_fact("Bangalore", "capital of", "Karnataka", fact_id=1)
        self.wm.add_original_fact("Karnataka", "is a state in", "India", fact_id=2)
        self.wm.add_original_fact("India", "located in", "Asia", fact_id=3)
        self.wm.add_original_fact("India", "borders", "Nepal", fact_id=4)
        self.wm.add_original_fact("Keyboard", "part of", "Console", fact_id=5)
        self.wm.add_original_fact("Console", "part of", "Workstation", fact_id=6)

        result = forward_chain(self.wm, rules=get_all_rules())

        self.assertEqual(result.status, "success")
        # Inferred facts expected:
        # R1: (Bangalore located in India)
        # R2 (via inferred fact): (Bangalore located in Asia)
        # R4: (Nepal borders India)
        # R3: (Keyboard part of Workstation)
        triples = [f.triple for f in result.inferred_facts]
        self.assertIn(("Bangalore", "located in", "India"), triples)
        self.assertIn(("Bangalore", "located in", "Asia"), triples)
        self.assertIn(("Nepal", "borders", "India"), triples)
        self.assertIn(("Keyboard", "part of", "Workstation"), triples)
        self.assertEqual(len(result.inferred_facts), 4)


if __name__ == '__main__':
    unittest.main()
