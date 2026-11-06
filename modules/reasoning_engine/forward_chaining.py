"""
Forward Chaining Module
Implements classical data-driven deduction (production system inference) for DocuMind AI.

ACADEMIC EXPLANATION:
In Symbolic Artificial Intelligence, Forward Chaining is a data-driven reasoning strategy:
1. Begin with known ground facts in Working Memory.
2. At each cycle, match active inference rules against all available facts (Rule Matcher).
3. For each successful match whose conclusion is not yet known (Modus Ponens),
   instantiate and derive the new inferred fact.
4. Add the inferred fact to Working Memory and log an explanatory InferenceTrace.
5. Inferred facts immediately become candidates to satisfy subsequent premises in future cycles.
6. Repeat until a deductive fixed point is reached (quiescence: no new facts can be derived)
   or a configured resource limit (max iterations / max facts) is encountered.
"""

from typing import List, Optional, Union, Tuple, Set
from .reasoning_models import Fact, InferenceRule, InferenceTrace, ReasoningResult
from .working_memory import WorkingMemory
from .rule_base import get_all_rules
from .rule_matcher import RuleMatcher, RuleMatch


class ForwardChainer:
    """
    Data-driven forward chaining inference engine.
    Executes iterative deductive cycles over WorkingMemory until reaching a fixed point.
    """

    def __init__(self,
                 rules: Optional[List[InferenceRule]] = None,
                 max_iterations: int = 10,
                 max_facts: int = 100):
        """
        Initialize the forward chaining inference engine.
        
        Args:
            rules: List of inference rules to apply (defaults to all rules in Rule Base).
            max_iterations: Safety threshold on derivation cycles.
            max_facts: Safety threshold on the maximum number of new inferred facts.
        """
        self.rules = list(rules) if rules is not None else get_all_rules()
        self.max_iterations = max(1, max_iterations)
        self.max_facts = max(1, max_facts)
        self.matcher = RuleMatcher()

    def run(self,
            working_memory: WorkingMemory,
            rules: Optional[List[InferenceRule]] = None) -> ReasoningResult:
        """
        Execute forward chaining inference over the provided WorkingMemory.
        
        Args:
            working_memory: The active in-memory Working Knowledge Base.
            rules: Optional override for the rule set to apply.
            
        Returns:
            ReasoningResult containing the execution status, all facts, and inference traces.
        """
        active_rules = list(rules) if rules is not None else self.rules
        traces: List[InferenceTrace] = []

        if working_memory is None:
            return ReasoningResult(
                status='error',
                explanation='WorkingMemory instance is required.'
            )

        # Check if memory is empty
        if working_memory.count_total() == 0 or not active_rules:
            return ReasoningResult(
                status='no_inference',
                original_facts=working_memory.get_original_facts(),
                inferred_facts=working_memory.get_inferred_facts(),
                inference_traces=[],
                explanation='No inference possible: working memory is empty or no rules provided.'
            )

        status = 'no_inference'
        iteration_count = 0
        total_new_facts = 0

        # Iterative derivation cycle until fixed point or limits reached
        while iteration_count < self.max_iterations:
            iteration_count += 1
            facts_added_this_iteration = 0

            # Match each rule against the current facts in Working Memory
            for rule in active_rules:
                matches = self.matcher.match_rule(rule, working_memory)

                for match in matches:
                    # 1. Instantiate conclusion triple
                    conclusion = match.conclusion_template
                    if not isinstance(conclusion, (tuple, list)) or len(conclusion) != 3:
                        continue

                    subj, pred, obj = conclusion
                    subj_str = str(subj).strip()
                    pred_str = str(pred).strip()
                    obj_str = str(obj).strip()

                    # 2. Reject meaningless self-relations (e.g. X located in X, X borders X)
                    if subj_str.lower() == obj_str.lower():
                        continue

                    # 3. Duplicate prevention: check if conclusion already exists in memory
                    if working_memory.contains_triple(subj_str, pred_str, obj_str):
                        continue

                    # 4. Check maximum facts safety threshold
                    if total_new_facts >= self.max_facts:
                        status = 'max_facts_reached'
                        break

                    # 5. Determine unique inferred fact ID and supporting IDs
                    inferred_counter = working_memory._inferred_counter
                    inferred_id = f"inf_{inferred_counter}"

                    supporting_ids = [
                        f.fact_id if f.fact_id is not None else f"{f.subject}-{f.predicate}-{f.object}"
                        for f in match.matching_facts
                    ]

                    # 6. Construct new inferred Fact object
                    new_fact = Fact(
                        subject=subj_str,
                        predicate=pred_str,
                        object=obj_str,
                        fact_type='inferred',
                        fact_id=inferred_id,
                        source_sentence=f"Inferred via Rule {rule.rule_id} ({rule.rule_name})"
                    )

                    # 7. Add inferred fact to WorkingMemory
                    if working_memory.add_fact(new_fact):
                        facts_added_this_iteration += 1
                        total_new_facts += 1

                        # 8. Record audit trace explaining the deduction
                        explanation = self._format_trace_explanation(rule, match, (subj_str, pred_str, obj_str))
                        trace = InferenceTrace(
                            derived_fact=new_fact,
                            applied_rule_id=rule.rule_id,
                            supporting_fact_ids=supporting_ids,
                            reasoning_method='forward_chaining',
                            explanation=explanation
                        )
                        traces.append(trace)

                if total_new_facts >= self.max_facts:
                    break

            if total_new_facts >= self.max_facts:
                status = 'max_facts_reached'
                break

            # Fixed point check: if no new facts were derived in this complete iteration, stop
            if facts_added_this_iteration == 0:
                if total_new_facts > 0:
                    status = 'success'
                else:
                    status = 'no_inference'
                break
        else:
            # Loop ended because max_iterations was hit without reaching quiescence
            if total_new_facts > 0:
                status = 'max_iterations_reached'
            else:
                status = 'no_inference'

        # Generate summary explanation
        if status == 'success':
            summary_exp = (f"Forward chaining completed successfully. Reached deductive fixed point "
                           f"after {iteration_count} iterations, deriving {total_new_facts} new facts.")
        elif status == 'no_inference':
            summary_exp = "No new facts could be inferred. Working memory is at a fixed point."
        elif status == 'max_iterations_reached':
            summary_exp = (f"Forward chaining halted: maximum iteration limit ({self.max_iterations}) "
                           f"reached with {total_new_facts} facts derived.")
        elif status == 'max_facts_reached':
            summary_exp = (f"Forward chaining halted: maximum inferred fact limit ({self.max_facts}) "
                           f"reached.")
        else:
            summary_exp = f"Forward chaining completed with status '{status}'."

        return ReasoningResult(
            status=status,
            original_facts=working_memory.get_original_facts(),
            inferred_facts=working_memory.get_inferred_facts(),
            inference_traces=traces,
            explanation=summary_exp
        )

    def _format_trace_explanation(self,
                                  rule: InferenceRule,
                                  match: RuleMatch,
                                  conclusion: Tuple[str, str, str]) -> str:
        """Construct a human-readable deductive explanation for an inference step."""
        subj, pred, obj = conclusion
        premises_text = " AND ".join(
            f"({f.subject} -[{f.predicate}]-> {f.object})" for f in match.matching_facts
        )
        return (f"Derived ({subj} -[{pred}]-> {obj}) via Rule {rule.rule_id} "
                f"[{rule.rule_name}] from premises: {premises_text}.")


def forward_chain(working_memory: WorkingMemory,
                  rules: Optional[List[InferenceRule]] = None,
                  max_iterations: int = 10,
                  max_facts: int = 100) -> ReasoningResult:
    """
    Convenience function to execute forward chaining on a WorkingMemory instance.
    """
    chainer = ForwardChainer(rules=rules, max_iterations=max_iterations, max_facts=max_facts)
    return chainer.run(working_memory)
