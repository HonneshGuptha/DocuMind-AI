"""
Backward Chaining Module
Implements classical goal-driven deduction (backward reasoning) for DocuMind AI.

ACADEMIC EXPLANATION:
In Symbolic Artificial Intelligence, Backward Chaining is a goal-driven reasoning strategy:
1. Begin with a hypothesis / target proposition (the Goal).
2. Check if the goal is directly satisfied by ground facts in Working Memory.
3. If not, search for inference rules whose conclusions unify with the goal (Modus Tollens / SLD resolution).
4. Substitute bound variables to generate intermediate Subgoals from the rule premises.
5. Recursively attempt to prove each subgoal.
6. A goal is proven if and only if all premises of at least one valid rule chain succeed.
7. If no supporting rules or ground facts can establish the goal, return unproven (Open-World Assumption:
   unproven does not imply false).
8. Working Memory remains strictly read-only throughout backward chaining.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Union, Tuple, Set, Dict, Any
from .reasoning_models import Fact, InferenceRule, InferenceTrace, ReasoningResult
from .working_memory import WorkingMemory
from .rule_base import get_all_rules
from .rule_matcher import is_variable, get_variable_name


class ProofStatus(str):
    """
    Subclass of str supporting academic status aliases:
    - 'proven' equals 'success'
    - 'unproven' equals 'unknown'
    """
    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, str):
            return False
        s = self.lower()
        o = other.lower()
        if s in ('proven', 'success') and o in ('proven', 'success'):
            return True
        if s in ('unproven', 'unknown') and o in ('unproven', 'unknown'):
            return True
        return s == o


@dataclass
class ProofResult(ReasoningResult):
    """
    Represents the structured outcome of a backward chaining goal proof.
    Inherits from ReasoningResult for full backward compatibility.
    
    Attributes:
        goal: The query proposition (subject, predicate, object).
        is_proven: True if the goal was established by ground facts or deduction.
        directly_found: True if the goal existed verbatim in WorkingMemory.
        applied_rule_ids: Ordered list of rule IDs fired along the proof path.
        supporting_fact_ids: List of fact IDs that ground the proof.
        subgoals: List of intermediate subgoals traversed during reasoning.
        reasoning_method: Always 'backward_chaining'.
    """
    goal: Tuple[str, str, str] = field(default_factory=tuple)
    is_proven: bool = False
    directly_found: bool = False
    applied_rule_ids: List[str] = field(default_factory=list)
    supporting_fact_ids: List[Union[int, str]] = field(default_factory=list)
    subgoals: List[Tuple[str, str, str]] = field(default_factory=list)
    reasoning_method: str = 'backward_chaining'

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            'goal': self.goal,
            'is_proven': self.is_proven,
            'directly_found': self.directly_found,
            'applied_rule_ids': list(self.applied_rule_ids),
            'supporting_fact_ids': list(self.supporting_fact_ids),
            'subgoals': list(self.subgoals),
            'reasoning_method': self.reasoning_method,
        })
        return base


class BackwardChainer:
    """
    Goal-driven backward chaining reasoning engine.
    Recursively proves target propositions against WorkingMemory and Rule Base.
    """

    def __init__(self,
                 rules: Optional[List[InferenceRule]] = None,
                 max_depth: int = 10,
                 max_evaluations: int = 100):
        """
        Initialize BackwardChainer.
        
        Args:
            rules: Available inference rules (defaults to Rule Base).
            max_depth: Maximum call stack recursion depth to avoid infinite loops.
            max_evaluations: Maximum rule evaluations / proof attempts.
        """
        self.rules = list(rules) if rules is not None else get_all_rules()
        self.max_depth = max(1, max_depth)
        self.max_evaluations = max(1, max_evaluations)
        self._evaluation_count = 0

    def prove(self,
              goal: Union[Tuple[str, str, str], List[str], Fact, Any],
              working_memory: WorkingMemory,
              rules: Optional[List[InferenceRule]] = None) -> ProofResult:
        """
        Attempt to prove a target goal proposition using backward chaining.
        
        Args:
            goal: Target proposition (subject, predicate, object).
            working_memory: The active in-memory Working Knowledge Base (read-only).
            rules: Optional override for active inference rules.
            
        Returns:
            ProofResult detailing success/failure, supporting facts, rule path, and traces.
        """
        active_rules = list(rules) if rules is not None else self.rules
        norm_goal = self._normalize_goal(goal)
        self._evaluation_count = 0

        if working_memory is None:
            return ProofResult(
                status=ProofStatus('error'),
                goal=norm_goal,
                is_proven=False,
                explanation='WorkingMemory instance is required.'
            )

        if not norm_goal:
            return ProofResult(
                status=ProofStatus('unproven'),
                goal=norm_goal,
                is_proven=False,
                explanation='Invalid or empty goal.'
            )

        # Execute recursive goal proof
        success, directly_found, supporting_facts, rule_ids, traces, subgoals, exit_status = self._prove_recursive(
            goal=norm_goal,
            working_memory=working_memory,
            rules=active_rules,
            active_goals=set(),
            current_depth=0
        )

        # Supporting fact IDs deduplicated while preserving order
        seen_fids = set()
        supporting_fids: List[Union[int, str]] = []
        for f in supporting_facts:
            fid = f.fact_id if f.fact_id is not None else f.triple
            if fid not in seen_fids:
                seen_fids.add(fid)
                supporting_fids.append(fid)

        if success:
            status = ProofStatus('proven')
            if directly_found:
                explanation = (f"Goal ({norm_goal[0]} -[{norm_goal[1]}]-> {norm_goal[2]}) "
                               f"is directly verified by ground fact in Working Memory.")
            else:
                rules_str = " -> ".join(rule_ids)
                explanation = (f"Goal ({norm_goal[0]} -[{norm_goal[1]}]-> {norm_goal[2]}) "
                               f"is proven via backward chaining using rules [{rules_str}] "
                               f"supported by facts {supporting_fids}.")
        else:
            status = ProofStatus(exit_status)
            if exit_status == 'cycle_detected':
                explanation = f"Goal ({norm_goal[0]} -[{norm_goal[1]}]-> {norm_goal[2]}) unproven: cyclic dependency detected."
            elif exit_status == 'max_depth_reached':
                explanation = f"Goal ({norm_goal[0]} -[{norm_goal[1]}]-> {norm_goal[2]}) unproven: maximum recursion depth exceeded."
            elif exit_status == 'max_evaluations_reached':
                explanation = f"Goal ({norm_goal[0]} -[{norm_goal[1]}]-> {norm_goal[2]}) unproven: evaluation limit exceeded."
            else:
                explanation = f"Goal ({norm_goal[0]} -[{norm_goal[1]}]-> {norm_goal[2]}) could not be proven from known facts or rules."

        return ProofResult(
            status=status,
            goal=norm_goal,
            is_proven=success,
            directly_found=directly_found,
            applied_rule_ids=rule_ids,
            supporting_fact_ids=supporting_fids,
            subgoals=subgoals,
            original_facts=working_memory.get_original_facts(),
            inferred_facts=[],
            inference_traces=traces,
            explanation=explanation,
            reasoning_method='backward_chaining'
        )

    def backward_chain(self,
                       goal: Any,
                       working_memory: WorkingMemory,
                       rules: Optional[List[InferenceRule]] = None) -> ProofResult:
        """Alias for prove."""
        return self.prove(goal, working_memory, rules)

    def _normalize_goal(self, goal: Any) -> Tuple[str, str, str]:
        """Normalize goal input into a canonical (subject, predicate, object) tuple."""
        if isinstance(goal, (tuple, list)) and len(goal) == 3:
            return (str(goal[0]).strip(), str(goal[1]).strip(), str(goal[2]).strip())
        if hasattr(goal, 'subject') and hasattr(goal, 'predicate') and hasattr(goal, 'object'):
            return (str(goal.subject).strip(), str(goal.predicate).strip(), str(goal.object).strip())
        return ()

    def _prove_recursive(self,
                         goal: Tuple[str, str, str],
                         working_memory: WorkingMemory,
                         rules: List[InferenceRule],
                         active_goals: Set[Tuple[str, str, str]],
                         current_depth: int) -> Tuple[bool, bool, List[Fact], List[str], List[InferenceTrace], List[Tuple[str, str, str]], str]:
        """
        Recursive SLD-style backward proving engine.
        Returns:
            (is_proven, directly_found, supporting_facts, rule_ids, traces, subgoals, exit_status)
        """
        g_subj, g_pred, g_obj = goal
        norm_key = (g_subj.lower(), g_pred.lower(), g_obj.lower())

        # 1. Cycle detection: if goal is currently an active ancestor on the call stack
        if norm_key in active_goals:
            return False, False, [], [], [], [], 'cycle_detected'

        # 2. Recursion depth limit check
        if current_depth >= self.max_depth:
            return False, False, [], [], [], [], 'max_depth_reached'

        # 3. Rule evaluation limit check
        self._evaluation_count += 1
        if self._evaluation_count > self.max_evaluations:
            return False, False, [], [], [], [], 'max_evaluations_reached'

        # 4. Direct ground fact lookup in Working Memory
        ground_matches = working_memory.search_facts(
            subject=g_subj,
            predicate=g_pred,
            object=g_obj,
            exact=True
        )
        if ground_matches:
            found_fact = ground_matches[0]
            trace = InferenceTrace(
                derived_fact=found_fact,
                applied_rule_id="DIRECT_LOOKUP",
                supporting_fact_ids=[found_fact.fact_id if found_fact.fact_id is not None else found_fact.triple],
                reasoning_method="backward_chaining",
                explanation=f"Directly verified from ground fact #{found_fact.fact_id}."
            )
            return True, True, [found_fact], [], [trace], [], 'proven'

        # Mark goal as active on call stack
        new_active = active_goals | {norm_key}
        accumulated_subgoals: List[Tuple[str, str, str]] = []
        last_failure_reason = 'unproven'

        # 5. Search for inference rules whose conclusions unify with this goal
        for rule in rules:
            conc = rule.conclusion
            if not isinstance(conc, (tuple, list)) or len(conc) != 3:
                continue
            r_s, r_p, r_o = conc

            # Check predicate match (case-insensitive)
            if r_p.strip().lower() != g_pred.strip().lower():
                continue

            # Unify conclusion with goal to form initial bindings
            init_bindings: Dict[str, str] = {}
            if is_variable(r_s):
                init_bindings[get_variable_name(r_s)] = g_subj
            elif r_s.strip().lower() != g_subj.strip().lower():
                continue

            if is_variable(r_o):
                init_bindings[get_variable_name(r_o)] = g_obj
            elif r_o.strip().lower() != g_obj.strip().lower():
                continue

            # Attempt to satisfy rule premises with initial bindings
            premises = rule.premises if isinstance(rule.premises, list) else [rule.premises]
            proof_ok, supp_facts, r_ids, trs, subg, reason = self._solve_premises(
                premises=premises,
                bindings=init_bindings,
                working_memory=working_memory,
                rules=rules,
                active_goals=new_active,
                current_depth=current_depth + 1
            )
            accumulated_subgoals.extend(subg)

            if proof_ok:
                derived_fact = Fact(
                    subject=g_subj,
                    predicate=g_pred,
                    object=g_obj,
                    fact_type='inferred',
                    fact_id=f"proof_{g_subj}_{g_pred}_{g_obj}",
                    source_sentence=f"Proven via backward chaining with Rule {rule.rule_id}"
                )
                rule_trace = InferenceTrace(
                    derived_fact=derived_fact,
                    applied_rule_id=rule.rule_id,
                    supporting_fact_ids=[f.fact_id if f.fact_id is not None else f.triple for f in supp_facts],
                    reasoning_method='backward_chaining',
                    explanation=f"Proved goal ({g_subj} {g_pred} {g_obj}) using Rule {rule.rule_id}."
                )
                return True, False, supp_facts, r_ids + [rule.rule_id], trs + [rule_trace], accumulated_subgoals, 'proven'
            else:
                if reason in ('cycle_detected', 'max_depth_reached', 'max_evaluations_reached'):
                    last_failure_reason = reason

        return False, False, [], [], [], accumulated_subgoals, last_failure_reason

    def _solve_premises(self,
                        premises: List[Any],
                        bindings: Dict[str, str],
                        working_memory: WorkingMemory,
                        rules: List[InferenceRule],
                        active_goals: Set[Tuple[str, str, str]],
                        current_depth: int) -> Tuple[bool, List[Fact], List[str], List[InferenceTrace], List[Tuple[str, str, str]], str]:
        """
        Recursively solve a list of premises under current variable bindings.
        Handles both fully bound premises and premises with intermediate variables.
        """
        if not premises:
            return True, [], [], [], [], 'proven'

        first_premise = premises[0]
        remaining_premises = premises[1:]

        p_s, p_p, p_o = first_premise
        pred_val = p_p.strip()

        # Determine whether subject is bound or unbound variable
        if is_variable(p_s):
            var_s = get_variable_name(p_s)
            if var_s in bindings:
                subj_val = bindings[var_s]
                is_s_unbound = False
            else:
                subj_val = p_s
                is_s_unbound = True
        else:
            subj_val = p_s
            is_s_unbound = False

        # Determine whether object is bound or unbound variable
        if is_variable(p_o):
            var_o = get_variable_name(p_o)
            if var_o in bindings:
                obj_val = bindings[var_o]
                is_o_unbound = False
            else:
                obj_val = p_o
                is_o_unbound = True
        else:
            obj_val = p_o
            is_o_unbound = False

        # Case A: Both subject and object are bound -> Prove directly as a subgoal
        if not is_s_unbound and not is_o_unbound:
            subgoal = (str(subj_val).strip(), pred_val, str(obj_val).strip())
            ok, _, facts, r_ids, trs, subg, reason = self._prove_recursive(
                goal=subgoal,
                working_memory=working_memory,
                rules=rules,
                active_goals=active_goals,
                current_depth=current_depth
            )
            all_subgoals = [subgoal] + subg

            if not ok:
                return False, [], [], [], all_subgoals, reason

            # Proceed to remaining premises
            rem_ok, rem_facts, rem_r_ids, rem_trs, rem_subg, rem_reason = self._solve_premises(
                premises=remaining_premises,
                bindings=bindings,
                working_memory=working_memory,
                rules=rules,
                active_goals=active_goals,
                current_depth=current_depth
            )
            all_subgoals.extend(rem_subg)

            if rem_ok:
                return True, facts + rem_facts, r_ids + rem_r_ids, trs + rem_trs, all_subgoals, 'proven'
            return False, [], [], [], all_subgoals, rem_reason

        # Case B: Subject is bound, Object is an unbound intermediate variable (e.g. Y in R1/R2)
        elif not is_s_unbound and is_o_unbound:
            var_name = get_variable_name(p_o)
            # Find candidate facts in WorkingMemory matching subject and predicate
            candidates = working_memory.search_facts(subject=str(subj_val).strip(), predicate=pred_val, exact=True)
            last_reason = 'unproven'

            for cand_fact in candidates:
                cand_val = cand_fact.object
                subgoal = (str(subj_val).strip(), pred_val, cand_val)
                new_b = dict(bindings)
                new_b[var_name] = cand_val

                cand_trace = InferenceTrace(
                    derived_fact=cand_fact,
                    applied_rule_id="DIRECT_LOOKUP",
                    supporting_fact_ids=[cand_fact.fact_id if cand_fact.fact_id is not None else cand_fact.triple],
                    reasoning_method="backward_chaining",
                    explanation=f"Directly verified ({subgoal[0]} {subgoal[1]} {subgoal[2]}) from ground fact #{cand_fact.fact_id}."
                )

                rem_ok, rem_facts, rem_r_ids, rem_trs, rem_subg, reason = self._solve_premises(
                    premises=remaining_premises,
                    bindings=new_b,
                    working_memory=working_memory,
                    rules=rules,
                    active_goals=active_goals,
                    current_depth=current_depth
                )
                if rem_ok:
                    return True, [cand_fact] + rem_facts, rem_r_ids, [cand_trace] + rem_trs, [subgoal] + rem_subg, 'proven'
                last_reason = reason

            return False, [], [], [], [], last_reason

        # Case C: Object is bound, Subject is an unbound intermediate variable
        elif is_s_unbound and not is_o_unbound:
            var_name = get_variable_name(p_s)
            candidates = working_memory.search_facts(object=str(obj_val).strip(), predicate=pred_val, exact=True)
            last_reason = 'unproven'

            for cand_fact in candidates:
                cand_val = cand_fact.subject
                subgoal = (cand_val, pred_val, str(obj_val).strip())
                new_b = dict(bindings)
                new_b[var_name] = cand_val

                cand_trace = InferenceTrace(
                    derived_fact=cand_fact,
                    applied_rule_id="DIRECT_LOOKUP",
                    supporting_fact_ids=[cand_fact.fact_id if cand_fact.fact_id is not None else cand_fact.triple],
                    reasoning_method="backward_chaining",
                    explanation=f"Directly verified ({subgoal[0]} {subgoal[1]} {subgoal[2]}) from ground fact #{cand_fact.fact_id}."
                )

                rem_ok, rem_facts, rem_r_ids, rem_trs, rem_subg, reason = self._solve_premises(
                    premises=remaining_premises,
                    bindings=new_b,
                    working_memory=working_memory,
                    rules=rules,
                    active_goals=active_goals,
                    current_depth=current_depth
                )
                if rem_ok:
                    return True, [cand_fact] + rem_facts, rem_r_ids, [cand_trace] + rem_trs, [subgoal] + rem_subg, 'proven'
                last_reason = reason

            return False, [], [], [], [], last_reason

        return False, [], [], [], [], 'unproven'


def backward_chain(goal: Any,
                   working_memory: WorkingMemory,
                   rules: Optional[List[InferenceRule]] = None,
                   max_depth: int = 10,
                   max_evaluations: int = 100) -> ProofResult:
    """
    Convenience function to prove a goal using Backward Chaining.
    """
    chainer = BackwardChainer(rules=rules, max_depth=max_depth, max_evaluations=max_evaluations)
    return chainer.prove(goal, working_memory)
