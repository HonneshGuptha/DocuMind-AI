"""
Unit tests for Module 4 Stage 2: Rule Base and Rule Matcher.
Verifies:
- All four deterministic rules (R1, R2, R3, R4) are properly defined.
- Rule lookup by ID.
- Single-premise matching (R4).
- Two-premise matching (R1, R2, R3).
- Correct X/Y/Z variable binding.
- Repeated-variable consistency.
- Multiple valid combinations.
- Case-insensitive matching.
- Preservation of original entity casing in bindings.
- Unmatched premises return empty list safely.
- Empty fact collections.
- Duplicate combination prevention.
- WorkingMemory remains completely unchanged after matching.
"""

import unittest

from modules.reasoning_engine.reasoning_models import Fact, InferenceRule
from modules.reasoning_engine.working_memory import WorkingMemory
from modules.reasoning_engine.rule_base import (
    R1,
    R2,
    R3,
    R4,
    R1_CAPITAL_LOCATION,
    R2_LOCATION_TRANSITIVITY,
    R3_PART_WHOLE_TRANSITIVITY,
    R4_BORDER_SYMMETRY,
    get_all_rules,
    get_rule_by_id,
)
from modules.reasoning_engine.rule_matcher import (
    RuleMatcher,
    RuleMatch,
    match_rule,
    is_variable,
    get_variable_name,
)


class TestRuleBase(unittest.TestCase):
    """Tests for the declarative Rule Base and rule retrieval."""

    def test_all_four_rules_defined(self):
        """Verify all 4 canonical rules exist, have unique IDs, premises, and conclusions."""
        rules = get_all_rules()
        self.assertEqual(len(rules), 4)

        rule_ids = [r.rule_id for r in rules]
        self.assertIn("R1", rule_ids)
        self.assertIn("R2", rule_ids)
        self.assertIn("R3", rule_ids)
        self.assertIn("R4", rule_ids)

        # R1: Capital Location Rule
        self.assertEqual(R1.rule_name, "Capital Location Rule")
        self.assertEqual(len(R1.premises), 2)
        self.assertEqual(R1.premises[0], ("X", "capital of", "Y"))
        self.assertEqual(R1.premises[1], ("Y", "is a state in", "Z"))
        self.assertEqual(R1.conclusion, ("X", "located in", "Z"))

        # R2: Location Transitivity Rule
        self.assertEqual(R2.rule_name, "Location Transitivity Rule")
        self.assertEqual(len(R2.premises), 2)
        self.assertEqual(R2.premises[0], ("X", "located in", "Y"))
        self.assertEqual(R2.premises[1], ("Y", "located in", "Z"))
        self.assertEqual(R2.conclusion, ("X", "located in", "Z"))

        # R3: Part-Whole Transitivity Rule
        self.assertEqual(R3.rule_name, "Part-Whole Transitivity Rule")
        self.assertEqual(len(R3.premises), 2)
        self.assertEqual(R3.premises[0], ("X", "part of", "Y"))
        self.assertEqual(R3.premises[1], ("Y", "part of", "Z"))
        self.assertEqual(R3.conclusion, ("X", "part of", "Z"))

        # R4: Border Symmetry Rule
        self.assertEqual(R4.rule_name, "Border Symmetry Rule")
        self.assertEqual(len(R4.premises), 1)
        self.assertEqual(R4.premises[0], ("X", "borders", "Y"))
        self.assertEqual(R4.conclusion, ("Y", "borders", "X"))

    def test_rule_lookup_by_id(self):
        """Verify get_rule_by_id with short IDs, full descriptive IDs, and case insensitivity."""
        self.assertIs(get_rule_by_id("R1"), R1)
        self.assertIs(get_rule_by_id("r1"), R1)
        self.assertIs(get_rule_by_id("R1_CAPITAL_LOCATION"), R1)
        self.assertIs(get_rule_by_id("capital_location"), R1)

        self.assertIs(get_rule_by_id("R2"), R2)
        self.assertIs(get_rule_by_id("r2_location_transitivity"), R2)

        self.assertIs(get_rule_by_id("R3"), R3)
        self.assertIs(get_rule_by_id("r3_part_whole_transitivity"), R3)

        self.assertIs(get_rule_by_id("R4"), R4)
        self.assertIs(get_rule_by_id("border_symmetry"), R4)

        # Non-existent or invalid lookups
        self.assertIsNone(get_rule_by_id("R99"))
        self.assertIsNone(get_rule_by_id(""))
        self.assertIsNone(get_rule_by_id(None))


class TestRuleMatcher(unittest.TestCase):
    """Tests for pattern matching and variable binding."""

    def setUp(self):
        self.matcher = RuleMatcher()

    def test_single_premise_matching_border_symmetry(self):
        """Verify matching single-premise rule R4 (Border Symmetry)."""
        facts = [
            Fact(subject="Canada", predicate="borders", object="United States", fact_id=1)
        ]
        matches = self.matcher.match_rule(R4, facts)

        self.assertEqual(len(matches), 1)
        m = matches[0]
        self.assertEqual(m.rule.rule_id, "R4")
        self.assertEqual(m.bindings, {"X": "Canada", "Y": "United States"})
        self.assertEqual(len(m.matching_facts), 1)
        self.assertEqual(m.matching_facts[0].subject, "Canada")
        self.assertEqual(m.conclusion_template, ("United States", "borders", "Canada"))

    def test_two_premise_matching_and_xyz_binding(self):
        """Verify matching two-premise rule R1 with X, Y, Z variable binding."""
        facts = [
            Fact(subject="Bangalore", predicate="capital of", object="Karnataka", fact_id=1),
            Fact(subject="Karnataka", predicate="is a state in", object="India", fact_id=2),
        ]
        matches = self.matcher.match_rule(R1, facts)

        self.assertEqual(len(matches), 1)
        m = matches[0]
        self.assertEqual(m.bindings["X"], "Bangalore")
        self.assertEqual(m.bindings["Y"], "Karnataka")
        self.assertEqual(m.bindings["Z"], "India")
        self.assertEqual(m.conclusion_template, ("Bangalore", "located in", "India"))
        self.assertEqual(len(m.matching_facts), 2)
        self.assertEqual(m.matching_facts[0].fact_id, 1)
        self.assertEqual(m.matching_facts[1].fact_id, 2)

    def test_location_transitivity_matching(self):
        """Verify matching R2 (Location Transitivity)."""
        facts = [
            Fact(subject="Manhattan", predicate="located in", object="New York City", fact_id=10),
            Fact(subject="New York City", predicate="located in", object="USA", fact_id=11),
        ]
        matches = self.matcher.match_rule(R2, facts)

        self.assertEqual(len(matches), 1)
        m = matches[0]
        self.assertEqual(m.bindings["X"], "Manhattan")
        self.assertEqual(m.bindings["Y"], "New York City")
        self.assertEqual(m.bindings["Z"], "USA")
        self.assertEqual(m.conclusion_template, ("Manhattan", "located in", "USA"))

    def test_part_whole_transitivity_matching(self):
        """Verify matching R3 (Part-Whole Transitivity)."""
        facts = [
            Fact(subject="CPU", predicate="part of", object="Motherboard", fact_id=20),
            Fact(subject="Motherboard", predicate="part of", object="Computer", fact_id=21),
        ]
        matches = self.matcher.match_rule(R3, facts)

        self.assertEqual(len(matches), 1)
        m = matches[0]
        self.assertEqual(m.bindings["X"], "CPU")
        self.assertEqual(m.bindings["Y"], "Motherboard")
        self.assertEqual(m.bindings["Z"], "Computer")
        self.assertEqual(m.conclusion_template, ("CPU", "part of", "Computer"))

    def test_repeated_variable_consistency(self):
        """Verify that when repeated variables differ, matching fails."""
        facts = [
            Fact(subject="Bangalore", predicate="capital of", object="Karnataka", fact_id=1),
            # Object of premise 1 is Karnataka, but subject of premise 2 is Maharashtra
            Fact(subject="Maharashtra", predicate="is a state in", object="India", fact_id=2),
        ]
        matches = self.matcher.match_rule(R1, facts)
        self.assertEqual(len(matches), 0)

    def test_multiple_valid_combinations(self):
        """Verify finding multiple distinct valid combinations across facts."""
        facts = [
            Fact(subject="Bangalore", predicate="capital of", object="Karnataka", fact_id=1),
            Fact(subject="Karnataka", predicate="is a state in", object="India", fact_id=2),
            Fact(subject="Chennai", predicate="capital of", object="Tamil Nadu", fact_id=3),
            Fact(subject="Tamil Nadu", predicate="is a state in", object="India", fact_id=4),
        ]
        matches = self.matcher.match_rule(R1, facts)

        self.assertEqual(len(matches), 2)
        conclusions = [m.conclusion_template for m in matches]
        self.assertIn(("Bangalore", "located in", "India"), conclusions)
        self.assertIn(("Chennai", "located in", "India"), conclusions)

    def test_case_insensitive_matching(self):
        """Verify matching succeeds regardless of casing differences."""
        facts = [
            Fact(subject="tokyo", predicate="CAPITAL OF", object="japan", fact_id=1),
            Fact(subject="JAPAN", predicate="IS A STATE IN", object="asia", fact_id=2),
        ]
        matches = self.matcher.match_rule(R1, facts)
        self.assertEqual(len(matches), 1)
        m = matches[0]
        self.assertEqual(m.bindings["X"], "tokyo")
        self.assertEqual(m.bindings["Y"], "japan")
        self.assertEqual(m.bindings["Z"], "asia")
        self.assertEqual(m.conclusion_template, ("tokyo", "located in", "asia"))

    def test_preservation_of_original_entity_casing(self):
        """Verify original casing of entities from facts is preserved in bindings."""
        facts = [
            Fact(subject="Bangalore", predicate="capital of", object="Karnataka", fact_id=1),
            Fact(subject="karnataka", predicate="is a state in", object="India", fact_id=2),
        ]
        matches = self.matcher.match_rule(R1, facts)
        self.assertEqual(len(matches), 1)
        m = matches[0]
        # X preserves Bangalore, Y preserves Karnataka (from first binding), Z preserves India
        self.assertEqual(m.bindings["X"], "Bangalore")
        self.assertEqual(m.bindings["Y"], "Karnataka")
        self.assertEqual(m.bindings["Z"], "India")

    def test_unmatched_premises(self):
        """Verify that when only some premises are satisfied, no match is produced."""
        facts = [
            Fact(subject="Bangalore", predicate="capital of", object="Karnataka", fact_id=1),
            # Missing second premise (is a state in)
        ]
        matches = self.matcher.match_rule(R1, facts)
        self.assertEqual(len(matches), 0)

    def test_empty_fact_collections(self):
        """Verify empty fact collections return empty list safely without errors."""
        self.assertEqual(self.matcher.match_rule(R1, []), [])
        self.assertEqual(self.matcher.match_rule(R1, WorkingMemory()), [])

    def test_duplicate_combination_prevention(self):
        """Verify that duplicate fact entries do not produce duplicate RuleMatch instances."""
        facts = [
            Fact(subject="Canada", predicate="borders", object="United States", fact_id=1),
            # Same logical proposition repeated
            Fact(subject="canada", predicate="borders", object="united states", fact_id=2),
        ]
        matches = self.matcher.match_rule(R4, facts)
        self.assertEqual(len(matches), 1)

    def test_working_memory_remains_unchanged_after_matching(self):
        """Verify WorkingMemory is strictly read-only during matching."""
        wm = WorkingMemory()
        wm.add_original_fact("Bangalore", "capital of", "Karnataka", fact_id=1)
        wm.add_original_fact("Karnataka", "is a state in", "India", fact_id=2)

        count_before = wm.count_total()
        orig_before = wm.count_original()
        inf_before = wm.count_inferred()

        # Execute rule matching on WorkingMemory instance
        matches = self.matcher.match_rule(R1, wm)
        self.assertEqual(len(matches), 1)

        # WorkingMemory state must remain 100% unaltered
        self.assertEqual(wm.count_total(), count_before)
        self.assertEqual(wm.count_original(), orig_before)
        self.assertEqual(wm.count_inferred(), inf_before)
        self.assertEqual(wm.count_inferred(), 0)

    def test_convenience_match_rule_function(self):
        """Verify module-level match_rule function."""
        facts = [
            Fact(subject="A", predicate="borders", object="B")
        ]
        matches = match_rule(R4, facts)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].conclusion_template, ("B", "borders", "A"))

    def test_rule_match_access_and_unpacking(self):
        """Verify RuleMatch tuple unpacking, dict access, and serialization."""
        facts = [Fact(subject="A", predicate="borders", object="B", fact_id=9)]
        m = self.matcher.match_rule(R4, facts)[0]

        # Unpacking
        bindings, matched_facts = m
        self.assertEqual(bindings["X"], "A")
        self.assertEqual(len(matched_facts), 1)

        # Dict / index access
        self.assertEqual(m["X"], "A")
        self.assertEqual(m["bindings"]["X"], "A")
        self.assertEqual(m["conclusion_template"], ("B", "borders", "A"))
        self.assertEqual(m[0], bindings)
        self.assertEqual(m[1], matched_facts)

        # Serialization
        d = m.to_dict()
        self.assertEqual(d["rule_id"], "R4")
        self.assertEqual(d["bindings"], {"X": "A", "Y": "B"})
        self.assertEqual(d["matching_fact_ids"], [9])
        self.assertEqual(d["conclusion_template"], ("B", "borders", "A"))


if __name__ == '__main__':
    unittest.main()
