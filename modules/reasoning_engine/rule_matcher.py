"""
Rule Matcher Module
Implements deterministic pattern matching and variable binding for classical AI reasoning.

ACADEMIC EXPLANATION:
In Symbolic Artificial Intelligence (e.g. Rete algorithm, unification in Prolog),
Rule Matching is the phase where rule premises are compared against the facts
currently present in Working Memory.

Variables (e.g. X, Y, Z) act as placeholders.
- A variable can bind to any entity upon first encounter.
- When the same variable appears multiple times across premises, it must bind to
  the identical entity (consistency condition).
- Matching is deterministic and case-insensitive.
- The original entity casing from the ground facts is strictly preserved in bindings.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional, Union, Set
from .reasoning_models import Fact, InferenceRule
from .working_memory import WorkingMemory


# Set of single-letter standard variable identifiers
_STANDARD_VARIABLES = {'X', 'Y', 'Z', 'W', 'V', 'U', 'A', 'B', 'C', 'P', 'Q'}


def is_variable(token: Any) -> bool:
    """
    Determine if a pattern element represents a logical variable.
    
    A token is recognized as a variable if:
    1. It starts with '?' (e.g. '?X', '?entity')
    2. It is a single uppercase letter (e.g. 'X', 'Y', 'Z')
    3. It is in the standard variable set.
    """
    if not isinstance(token, str):
        return False
    t = token.strip()
    if t.startswith('?'):
        return len(t) > 1 and t[1:].isalnum()
    return t in _STANDARD_VARIABLES or (len(t) == 1 and t.isupper())


def get_variable_name(token: str) -> str:
    """Extract canonical variable name by stripping leading '?' if present."""
    t = token.strip()
    if t.startswith('?'):
        return t[1:]
    return t


@dataclass
class RuleMatch:
    """
    Represents a successful match of an InferenceRule against a specific set of Facts.
    
    Attributes:
        rule: The InferenceRule that matched.
        bindings: Map from variable names to bound entity strings (e.g. {'X': 'Bangalore'}).
        matching_facts: Ordered list of Fact instances satisfying the premises.
    """
    rule: InferenceRule
    bindings: Dict[str, str]
    matching_facts: List[Fact]

    @property
    def conclusion_template(self) -> Tuple[str, str, str]:
        """
        Instantiate the rule's conclusion template using the variable bindings.
        E.g. for conclusion ('X', 'located in', 'Z') with bindings
        {'X': 'Bangalore', 'Z': 'India'}, returns ('Bangalore', 'located in', 'India').
        """
        conc = self.rule.conclusion
        if isinstance(conc, (list, tuple)) and len(conc) == 3:
            s = self._substitute_term(conc[0])
            p = self._substitute_term(conc[1])
            o = self._substitute_term(conc[2])
            return (s, p, o)
        return conc

    def _substitute_term(self, term: Any) -> Any:
        if isinstance(term, str):
            t_clean = term.strip()
            var = get_variable_name(t_clean)
            if var in self.bindings:
                return self.bindings[var]
            if t_clean in self.bindings:
                return self.bindings[t_clean]
        return term

    def __iter__(self):
        """Allows tuple unpacking: bindings, facts = match"""
        return iter((self.bindings, self.matching_facts))

    def __getitem__(self, item):
        """Allows dictionary-style or indexed access to match properties."""
        if item in self.bindings:
            return self.bindings[item]
        if item == 'bindings':
            return self.bindings
        if item in ('matching_facts', 'facts'):
            return self.matching_facts
        if item == 'conclusion_template':
            return self.conclusion_template
        if item == 'rule':
            return self.rule
        if isinstance(item, int):
            return (self.bindings, self.matching_facts)[item]
        raise KeyError(item)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize match outcome to dictionary."""
        return {
            'rule_id': self.rule.rule_id,
            'bindings': dict(self.bindings),
            'matching_fact_ids': [f.fact_id for f in self.matching_facts],
            'conclusion_template': self.conclusion_template
        }

    def __repr__(self) -> str:
        return (f"<RuleMatch Rule={self.rule.rule_id} "
                f"bindings={self.bindings} conclusion={self.conclusion_template}>")


class RuleMatcher:
    """
    Deterministic rule matcher for classical rule-based AI reasoning.
    Finds all valid variable substitutions and supporting fact sets for a given InferenceRule.
    """

    def match_rule(self,
                   rule: InferenceRule,
                   facts: Union[List[Fact], WorkingMemory, Any]) -> List[RuleMatch]:
        """
        Match an InferenceRule against a collection of facts.
        
        Args:
            rule: The InferenceRule containing premises to match.
            facts: List of Fact objects or a WorkingMemory instance.
            
        Returns:
            List of unique RuleMatch instances. Returns empty list if no matches found.
        """
        # Extract facts list from WorkingMemory or collection safely
        if isinstance(facts, WorkingMemory):
            fact_list = facts.get_all_facts()
        elif hasattr(facts, 'get_all_facts'):
            fact_list = facts.get_all_facts()
        elif hasattr(facts, '__iter__'):
            fact_list = list(facts)
        else:
            return []

        if not fact_list or not rule:
            return []

        # Normalize premises into a list of 3-tuples
        premises = self._normalize_premises(rule)
        if not premises:
            return []

        # Recursively search for satisfying fact combinations
        raw_matches: List[Tuple[Dict[str, str], List[Fact]]] = []
        self._find_matches_recursive(
            premises=premises,
            facts=fact_list,
            current_bindings={},
            matched_facts=[],
            used_fact_indices=set(),
            results=raw_matches
        )

        # Eliminate duplicate combinations
        seen_keys: Set[Tuple[Tuple[Tuple[str, str], ...], Tuple[Tuple[str, str, str], ...]]] = set()
        unique_matches: List[RuleMatch] = []

        for bindings, matched_facts in raw_matches:
            # Create a normalized deduplication signature
            bindings_sig = tuple(sorted((k, v.strip().lower()) for k, v in bindings.items()))
            facts_sig = tuple(f.normalized_key for f in matched_facts)
            combo_sig = (bindings_sig, facts_sig)

            if combo_sig not in seen_keys:
                seen_keys.add(combo_sig)
                unique_matches.append(
                    RuleMatch(
                        rule=rule,
                        bindings=bindings,
                        matching_facts=matched_facts
                    )
                )

        return unique_matches

    def match(self,
              rule: InferenceRule,
              facts: Union[List[Fact], WorkingMemory, Any]) -> List[RuleMatch]:
        """Convenience alias for match_rule."""
        return self.match_rule(rule, facts)

    def _normalize_premises(self, rule: InferenceRule) -> List[Tuple[Any, Any, Any]]:
        """Normalize premises to a list of (subject, predicate, object) tuples."""
        premises = rule.premises
        if isinstance(premises, tuple) and len(premises) == 3 and isinstance(premises[1], str):
            return [premises]
        return list(premises)

    def _find_matches_recursive(self,
                                premises: List[Tuple[Any, Any, Any]],
                                facts: List[Fact],
                                current_bindings: Dict[str, str],
                                matched_facts: List[Fact],
                                used_fact_indices: Set[int],
                                results: List[Tuple[Dict[str, str], List[Fact]]]) -> None:
        """
        Recursive backtracking unification matcher across premises.
        """
        if not premises:
            # All premises satisfied!
            results.append((current_bindings, matched_facts))
            return

        current_premise = premises[0]
        remaining_premises = premises[1:]
        p_sub, p_pred, p_obj = current_premise

        for idx, fact in enumerate(facts):
            # A distinct premise should match a distinct fact in the chain
            if idx in used_fact_indices:
                continue

            # 1. Match Predicate
            pred_ok, b_pred = self._match_term(p_pred, fact.predicate, current_bindings)
            if not pred_ok:
                continue

            # 2. Match Subject
            sub_ok, b_sub = self._match_term(p_sub, fact.subject, b_pred)
            if not sub_ok:
                continue

            # 3. Match Object
            obj_ok, b_obj = self._match_term(p_obj, fact.object, b_sub)
            if not obj_ok:
                continue

            # Advance to the next premise with updated bindings and matched fact
            self._find_matches_recursive(
                premises=remaining_premises,
                facts=facts,
                current_bindings=b_obj,
                matched_facts=matched_facts + [fact],
                used_fact_indices=used_fact_indices | {idx},
                results=results
            )

    def _match_term(self,
                    pattern_term: Any,
                    fact_term: str,
                    bindings: Dict[str, str]) -> Tuple[bool, Dict[str, str]]:
        """
        Matches a single pattern term (variable or constant) against a fact term.
        
        Rules:
        - If variable and unbound: bind to fact_term (preserving original casing).
        - If variable and bound: check if fact_term equals bound value case-insensitively.
        - If constant: compare case-insensitively with fact_term.
        
        Returns:
            (True, updated_bindings) on match, or (False, original_bindings) on mismatch.
        """
        if is_variable(pattern_term):
            var = get_variable_name(pattern_term)
            if var in bindings:
                # Variable already bound: must match consistently (case-insensitive)
                existing_val = bindings[var]
                if existing_val.strip().lower() == fact_term.strip().lower():
                    return True, bindings
                else:
                    return False, bindings
            else:
                # Unbound variable: bind value preserving original entity casing
                new_bindings = dict(bindings)
                new_bindings[var] = fact_term.strip()
                return True, new_bindings
        else:
            # Constant term: compare case-insensitively
            p_str = str(pattern_term).strip().lower()
            f_str = str(fact_term).strip().lower()
            if p_str == f_str:
                return True, bindings
            else:
                return False, bindings


def match_rule(rule: InferenceRule,
               facts: Union[List[Fact], WorkingMemory, Any]) -> List[RuleMatch]:
    """
    Convenience function to match an InferenceRule against facts.
    """
    matcher = RuleMatcher()
    return matcher.match_rule(rule, facts)
