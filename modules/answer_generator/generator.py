"""
Answer Generator Module
Implements deterministic, template-based answer synthesis for DocuMind AI.

ACADEMIC EXPLANATION:
In Symbolic Artificial Intelligence, Answer Generation renders formal logical
propositions into natural language sentences using predefined syntactic templates.

Strict Academic Constraints:
- Zero NLP libraries, zero LLMs, zero vector embeddings, zero generative hallucinations.
- The answer text is built purely from verified ground facts or derived inference rules.
- Under the Open-World Assumption, absence of proof produces an 'unknown' response, never a false 'no'.
- If evidence is conflicting, all conflicting sources are disclosed without inventing a resolution.
"""

import html
import re
from typing import List, Dict, Any, Optional, Union, Tuple, Set
from modules.question_processor.query_models import StructuredQuery, RetrievalResult
from modules.question_processor.parser import QuestionParser
from modules.reasoning_engine.reasoning_models import Fact as ReasoningFact, ReasoningResult, InferenceTrace
from modules.reasoning_engine.backward_chaining import ProofResult
from .answer_models import AnswerResponse


# Human-friendly rule names
RULE_DISPLAY_NAMES = {
    'R1': 'Capital Location Rule',
    'R1_CAPITAL_LOCATION': 'Capital Location Rule',
    'R2': 'Location Transitivity Rule',
    'R2_LOCATION_TRANSITIVITY': 'Location Transitivity Rule',
    'R3': 'Part-Whole Transitivity Rule',
    'R3_PART_WHOLE_TRANSITIVITY': 'Part-Whole Transitivity Rule',
    'R4': 'Border Symmetry Rule',
    'R4_BORDER_SYMMETRY': 'Border Symmetry Rule',
}


class AnswerGenerator:
    """
    Deterministic template-based answer synthesizer.
    Translates retrieval and reasoning outcomes into human-readable answers.
    """

    def __init__(self):
        self.parser = QuestionParser()

    def generate_answer(self,
                        query: Union[str, StructuredQuery],
                        retrieval_result: Optional[RetrievalResult] = None,
                        reasoning_result: Optional[Union[ReasoningResult, ProofResult]] = None) -> AnswerResponse:
        """
        Synthesize an AnswerResponse from query parameters, retrieved facts, and reasoning traces.
        
        Args:
            query: Either a raw question string or a parsed StructuredQuery.
            retrieval_result: Optional RetrievalResult from QuestionRetriever.
            reasoning_result: Optional ReasoningResult or ProofResult from reasoning engine.
            
        Returns:
            AnswerResponse object containing answer_text, status, type, and provenance.
        """
        # 1. Normalize query input
        if isinstance(query, str):
            structured_query = self.parser.parse(query)
        elif isinstance(query, StructuredQuery):
            structured_query = query
        else:
            return AnswerResponse(
                original_question=str(query),
                answer_text="An error occurred: invalid question input.",
                answer_status="error",
                answer_type="none",
                confidence="none",
                explanation="Input query was neither a string nor a StructuredQuery instance."
            )

        original_question = structured_query.original_question

        # 2. Case F: Unsupported Question Check
        if structured_query.parsing_status != 'success':
            return AnswerResponse(
                original_question=original_question,
                answer_text="This question format is not currently supported by DocuMind AI.",
                answer_status="unsupported",
                answer_type="none",
                confidence="none",
                explanation=structured_query.error_message or "Question format could not be mapped to supported templates."
            )

        # 3. Extract and normalize supporting facts and reasoning outcome
        direct_facts = self._extract_facts_from_retrieval(retrieval_result)
        inferred_facts, proof_traces, applied_rules, is_proven_goal, is_directly_found_goal = self._extract_reasoning_info(reasoning_result)

        has_direct = len(direct_facts) > 0 or is_directly_found_goal
        has_inferred = len(inferred_facts) > 0 or (is_proven_goal and not is_directly_found_goal)

        # 4. Case E: Unknown Answer Check (No facts and no proof)
        if not has_direct and not has_inferred:
            return AnswerResponse(
                original_question=original_question,
                answer_text="I could not find sufficient information in the knowledge base to answer this question.",
                answer_status="unknown",
                answer_type="none",
                confidence="none",
                explanation="No matching facts or valid deductive inferences were found in the knowledge base."
            )

        # 5. Determine Answer Type
        if has_direct and has_inferred:
            answer_type = "mixed"
            confidence = "verified"
        elif has_inferred:
            answer_type = "inferred"
            confidence = "deduced"
        else:
            answer_type = "direct"
            confidence = "verified"

        # 6. Collate supporting facts and metadata
        all_fact_objects = direct_facts + inferred_facts
        supporting_facts_data = [self._fact_to_dict(f) for f in all_fact_objects]
        supporting_fact_ids = self._collect_fact_ids(all_fact_objects, reasoning_result)
        sources = self._collect_sources(all_fact_objects)

        # 7. Check for conflicting facts
        conflict_text = self._detect_conflicts(structured_query, direct_facts)
        if conflict_text:
            return AnswerResponse(
                original_question=original_question,
                answer_text=conflict_text,
                answer_status="answered",
                answer_type=answer_type,
                supporting_facts=supporting_facts_data,
                supporting_fact_ids=supporting_fact_ids,
                applied_rule_ids=applied_rules,
                explanation="Conflicting factual assertions were discovered in the knowledge base.",
                confidence="verified",
                sources=sources
            )

        # 8. Synthesize Answer Text using Deterministic Templates
        answer_text = self._render_template(
            query=structured_query,
            direct_facts=direct_facts,
            inferred_facts=inferred_facts,
            reasoning_result=reasoning_result,
            is_inferred=has_inferred and not has_direct
        )

        # 9. Build Explanation
        explanation = self._build_explanation(
            answer_type=answer_type,
            direct_facts=direct_facts,
            applied_rules=applied_rules,
            proof_traces=proof_traces,
            reasoning_result=reasoning_result
        )

        return AnswerResponse(
            original_question=original_question,
            answer_text=answer_text,
            answer_status="answered",
            answer_type=answer_type,
            supporting_facts=supporting_facts_data,
            supporting_fact_ids=supporting_fact_ids,
            applied_rule_ids=applied_rules,
            explanation=explanation,
            confidence=confidence,
            sources=sources
        )

    # ==========================================================================
    # Template Rendering Subsystem
    # ==========================================================================

    def _render_template(self,
                         query: StructuredQuery,
                         direct_facts: List[Any],
                         inferred_facts: List[Any],
                         reasoning_result: Optional[Union[ReasoningResult, ProofResult]],
                         is_inferred: bool) -> str:
        """Render natural language answer text based on question type and facts."""
        q_type = query.question_type
        expected_field = query.expected_answer_field

        # Primary source fact
        all_facts = direct_facts + inferred_facts
        primary_fact = all_facts[0] if all_facts else None

        # Handle ProofResult without explicit direct_facts list
        if not primary_fact and isinstance(reasoning_result, ProofResult) and reasoning_result.is_proven:
            s, p, o = reasoning_result.goal
            # Construct synthetic display representation
            class SyntheticFact:
                def __init__(self, s, p, o):
                    self.subject = s
                    self.predicate = p
                    self.object = o
            primary_fact = SyntheticFact(s, p, o)
            all_facts = [primary_fact]

        if not primary_fact:
            return "I could not find sufficient information in the knowledge base to answer this question."

        subj = primary_fact.subject
        pred = primary_fact.predicate
        obj = primary_fact.object

        # ----------------------------------------------------------------------
        # Case D: Boolean Questions
        # ----------------------------------------------------------------------
        if expected_field == 'boolean':
            # e.g. "Does Canada border United States?" -> "Yes, Canada borders United States."
            # e.g. "Is Bangalore located in India?" -> "Yes, Bangalore is located in India."
            is_speaker_query = query.subject == 'Speaker' or (isinstance(query.original_question, str) and re.search(r'\b(am\s+i|do\s+i)\b', query.original_question, re.I))
            if is_speaker_query:
                if pred.lower() == 'is a':
                    article = 'an' if obj and obj[0].lower() in 'aeiou' else 'a'
                    # If obj already starts with an article, don't double it
                    if obj.lower().startswith(('a ', 'an ')):
                        return f"Yes, you are {obj}."
                    return f"Yes, you are {article} {obj}."
                elif pred.startswith(('is ', 'was ')):
                    return f"Yes, you {pred} {obj}."
                else:
                    return f"Yes, you {pred} {obj}."

            if pred.lower() == 'capital of':
                return f"Yes, {subj} is the capital of {obj}."
            elif pred.lower() == 'is a':
                article = 'an' if obj and obj[0].lower() in 'aeiou' else 'a'
                if obj.lower().startswith(('a ', 'an ')):
                    return f"Yes, {subj} is {obj}."
                return f"Yes, {subj} is {article} {obj}."
            elif ' ' in pred and not pred.startswith(('is ', 'was ')):
                return f"Yes, {subj} is {pred} {obj}."
            elif pred.startswith(('is ', 'was ')):
                return f"Yes, {subj} {pred} {obj}."
            else:
                return f"Yes, {subj} {pred} {obj}."

        # ----------------------------------------------------------------------
        # Category 0: Name & Speaker Identity
        # ----------------------------------------------------------------------
        is_speaker_query = query.subject == 'Speaker' or (isinstance(query.original_question, str) and re.search(r'\b(my\s+name|who\s+am\s+i|where\s+do\s+i|where\s+i\s+live|do\s+i\s+study|what\s+am\s+i|where\s+do\s+i\s+work)\b', query.original_question, re.I))

        if q_type == 'name':
            if is_speaker_query:
                return f"Your name is {obj}."
            return f"The name of {subj} is {obj}."

        # ----------------------------------------------------------------------
        # Category 0b: Residence / Living Location
        # ----------------------------------------------------------------------
        elif q_type == 'residence':
            if is_speaker_query:
                return f"You live in {obj}."
            return f"{subj} lives in {obj}."

        # ----------------------------------------------------------------------
        # Category 0c: Employment / Workplace
        # ----------------------------------------------------------------------
        elif q_type == 'employment':
            if expected_field == 'subject':
                # "Who works at KPMG?" -> "Honnesh works at KPMG."
                return f"{subj} works at {obj}."
            if is_speaker_query:
                return f"You work at {obj}."
            return f"{subj} works at {obj}."

        # ----------------------------------------------------------------------
        # Category 0d: Education / Studies
        # ----------------------------------------------------------------------
        elif q_type == 'education':
            if is_speaker_query:
                if pred.lower() == 'is a':
                    return f"You are studying {obj}." if 'student' not in obj.lower() else f"You are an {obj}."
                return f"You study at {obj}."
            if pred.lower() == 'is a':
                return f"{subj} is an {obj}."
            return f"{subj} studies at {obj}."

        # ----------------------------------------------------------------------
        # Category 1: Capital
        # ----------------------------------------------------------------------
        if q_type == 'capital':
            if expected_field == 'subject':
                # "What is the capital of Karnataka?" -> "The capital of Karnataka is Bangalore."
                return f"The capital of {obj} is {subj}."
            else:
                # "What is Bangalore the capital of?" -> "Bangalore is the capital of Karnataka."
                return f"{subj} is the capital of {obj}."

        # ----------------------------------------------------------------------
        # Category 2: Location
        # ----------------------------------------------------------------------
        elif q_type == 'location':
            # "Where is Tokyo located?" -> "Tokyo is located in Japan."
            if len(all_facts) > 1:
                locations = [f.object for f in all_facts]
                loc_list = ", ".join(locations)
                return f"{subj} is located in: {loc_list}."
            return f"{subj} is located in {obj}."

        # ----------------------------------------------------------------------
        # Category 3: Establishment
        # ----------------------------------------------------------------------
        elif q_type == 'establishment':
            # "When was Stanford established?" -> "Stanford was established in 1885."
            if pred.startswith(('was ', 'is ')):
                return f"{subj} {pred} {obj}."
            return f"{subj} was {pred} {obj}."

        # ----------------------------------------------------------------------
        # Category 4: Creator / Author / Inventor
        # ----------------------------------------------------------------------
        elif q_type == 'creator':
            # "Who developed Python?" -> "Python was developed by Guido van Rossum."
            if expected_field == 'subject':
                return f"{obj} was {pred} {subj}."
            if pred.startswith(('was ', 'is ')):
                return f"{subj} {pred} {obj}."
            return f"{subj} was {pred} {obj}."

        # ----------------------------------------------------------------------
        # Category 5: Part-Whole
        # ----------------------------------------------------------------------
        elif q_type == 'part_whole':
            # "What is Piston part of?" -> "Piston is part of Engine."
            if pred.startswith(('is ', 'was ')):
                return f"{subj} {pred} {obj}."
            return f"{subj} is part of {obj}."

        # ----------------------------------------------------------------------
        # Category 6: Border
        # ----------------------------------------------------------------------
        elif q_type == 'border':
            # "What does Canada border?"
            if len(all_facts) > 1:
                borders = [f.object for f in all_facts]
                b_list = ", ".join(borders)
                return f"{subj} borders: {b_list}."
            return f"{subj} borders {obj}."

        # ----------------------------------------------------------------------
        # Category 7: Production
        # ----------------------------------------------------------------------
        elif q_type == 'production':
            if len(all_facts) > 1:
                items = [f.object for f in all_facts]
                return f"{subj} produces: {', '.join(items)}."
            return f"{subj} produces {obj}."

        # ----------------------------------------------------------------------
        # Category 8: Identity
        # ----------------------------------------------------------------------
        elif q_type == 'identity':
            # "Who is Guido van Rossum?"
            if subj.lower() == query.subject.lower() if query.subject else False:
                if pred.startswith(('is ', 'was ')):
                    return f"{subj} {pred} {obj}."
                return f"{subj} is {pred} {obj}."
            else:
                # Entity matched as object: e.g. (Python, developed by, Guido van Rossum)
                if pred.startswith(('was ', 'is ')):
                    return f"{subj} {pred} {obj}."
                return f"{subj} was {pred} {obj}."

        # ----------------------------------------------------------------------
        # Category 9: Classification
        # ----------------------------------------------------------------------
        elif q_type == 'classification':
            # "What is Python?" -> "Python is a programming language."
            if pred.lower() == 'is a':
                return f"{subj} is a {obj}."
            elif pred.lower() in ('part of', 'is part of'):
                return f"{subj} is part of {obj}."
            elif pred.startswith(('is ', 'was ')):
                return f"{subj} {pred} {obj}."
            return f"{subj} is a {obj}."

        # Generic fallback template
        if pred.startswith(('is ', 'was ')):
            return f"{subj} {pred} {obj}."
        return f"{subj} {pred} {obj}."

    # ==========================================================================
    # Conflict Detection Subsystem
    # ==========================================================================

    def _detect_conflicts(self, query: StructuredQuery, facts: List[Any]) -> Optional[str]:
        """
        Detect whether multiple retrieved facts provide conflicting answers
        for single-valued factual queries (e.g. capital, founding year).
        """
        if len(facts) <= 1:
            return None

        # Check single-valued relations: capital, establishment year
        if query.question_type in ('capital', 'establishment'):
            unique_objects = set()
            for f in facts:
                unique_objects.add(str(f.object).strip().lower())

            if len(unique_objects) > 1:
                conflict_items = []
                for f in facts:
                    fid = getattr(f, 'fact_id', '?')
                    conflict_items.append(f"{f.subject} was {f.predicate} {f.object} (Fact #{fid})")
                return f"Conflicting information found in the knowledge base: {'; '.join(conflict_items)}."

        return None

    # ==========================================================================
    # Helper & Provenance Extraction Methods
    # ==========================================================================

    def _extract_facts_from_retrieval(self, retrieval_result: Optional[RetrievalResult]) -> List[Any]:
        """Safely extract facts from RetrievalResult."""
        if retrieval_result and hasattr(retrieval_result, 'facts') and retrieval_result.status == 'success':
            return list(retrieval_result.facts)
        return []

    def _extract_reasoning_info(self, reasoning_result: Optional[Union[ReasoningResult, ProofResult]]) -> Tuple[List[Any], List[InferenceTrace], List[str], bool, bool]:
        """
        Extract derived facts, traces, rules, and proof status from reasoning results.
        Returns:
            (inferred_facts, traces, applied_rule_ids, is_proven_goal, is_directly_found_goal)
        """
        inferred_facts: List[Any] = []
        traces: List[InferenceTrace] = []
        applied_rules: List[str] = []
        is_proven_goal = False
        is_directly_found = False

        if reasoning_result is None:
            return inferred_facts, traces, applied_rules, is_proven_goal, is_directly_found

        if isinstance(reasoning_result, ProofResult):
            is_proven_goal = bool(reasoning_result.is_proven)
            is_directly_found = bool(reasoning_result.directly_found)
            applied_rules = list(reasoning_result.applied_rule_ids)
            traces = list(reasoning_result.inference_traces)

        elif isinstance(reasoning_result, ReasoningResult):
            inferred_facts = list(reasoning_result.inferred_facts)
            traces = list(reasoning_result.inference_traces)
            for t in traces:
                if t.applied_rule_id and t.applied_rule_id not in applied_rules:
                    applied_rules.append(t.applied_rule_id)
            if reasoning_result.status == 'success' and len(inferred_facts) > 0:
                is_proven_goal = True

        return inferred_facts, traces, applied_rules, is_proven_goal, is_directly_found

    def _fact_to_dict(self, fact: Any) -> Dict[str, Any]:
        """Serialize a Fact instance to dictionary safely."""
        return {
            'subject': getattr(fact, 'subject', ''),
            'predicate': getattr(fact, 'predicate', ''),
            'object': getattr(fact, 'object', ''),
            'fact_id': getattr(fact, 'fact_id', None),
            'document_id': getattr(fact, 'document_id', None),
            'source_sentence': getattr(fact, 'source_sentence', None),
            'sentence_index': getattr(fact, 'sentence_index', None),
            'fact_type': getattr(fact, 'fact_type', 'original')
        }

    def _collect_fact_ids(self, facts: List[Any], reasoning_result: Optional[Any]) -> List[Union[int, str]]:
        """Collect all fact IDs supporting the answer."""
        ids: List[Union[int, str]] = []
        seen = set()

        for f in facts:
            fid = getattr(f, 'fact_id', None)
            if fid is not None and fid not in seen:
                seen.add(fid)
                ids.append(fid)

        if isinstance(reasoning_result, ProofResult):
            for fid in reasoning_result.supporting_fact_ids:
                if fid not in seen:
                    seen.add(fid)
                    ids.append(fid)

        return ids

    def _collect_sources(self, facts: List[Any]) -> List[Dict[str, Any]]:
        """Collect verified source document IDs and sentences."""
        sources: List[Dict[str, Any]] = []
        seen = set()

        for f in facts:
            doc_id = getattr(f, 'document_id', None)
            sentence = getattr(f, 'source_sentence', None)
            fid = getattr(f, 'fact_id', None)

            if doc_id is not None or sentence is not None:
                key = (doc_id, sentence)
                if key not in seen:
                    seen.add(key)
                    sources.append({
                        'document_id': doc_id,
                        'source_sentence': sentence,
                        'fact_id': fid
                    })

        return sources

    def _build_explanation(self,
                           answer_type: str,
                           direct_facts: List[Any],
                           applied_rules: List[str],
                           proof_traces: List[InferenceTrace],
                           reasoning_result: Optional[Any]) -> str:
        """Construct a provenance explanation audit trail."""
        if answer_type == 'direct':
            if direct_facts:
                f = direct_facts[0]
                fid = getattr(f, 'fact_id', '?')
                sent = getattr(f, 'source_sentence', None)
                if sent:
                    return f"Directly verified from Knowledge Base Fact #{fid}: \"{sent}\""
                return f"Directly verified from Knowledge Base Fact #{fid}."
            return "Directly verified from ground knowledge base assertions."

        elif answer_type == 'inferred':
            rule_names = [f"{r} ({RULE_DISPLAY_NAMES.get(r, r)})" for r in applied_rules]
            rules_str = ", ".join(rule_names) if rule_names else "symbolic inference"

            supp_ids = []
            if isinstance(reasoning_result, ProofResult):
                supp_ids = reasoning_result.supporting_fact_ids
            elif proof_traces:
                for t in proof_traces:
                    supp_ids.extend(t.supporting_fact_ids)
                supp_ids = list(dict.fromkeys(supp_ids))

            supp_str = f" from supporting facts {supp_ids}" if supp_ids else ""
            return f"Inferred via {rules_str}{supp_str}."

        elif answer_type == 'mixed':
            return (f"Synthesized from a combination of direct ground facts and "
                    f"inferred conclusions using rules {applied_rules}.")

        return "No explanation available."


def generate_answer(query: Union[str, StructuredQuery],
                    retrieval_result: Optional[RetrievalResult] = None,
                    reasoning_result: Optional[Union[ReasoningResult, ProofResult]] = None) -> AnswerResponse:
    """Convenience helper function to generate an answer response."""
    generator = AnswerGenerator()
    return generator.generate_answer(query, retrieval_result, reasoning_result)
