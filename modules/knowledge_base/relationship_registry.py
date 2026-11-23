"""
Relationship Registry Module
Defines canonical relationships, logical properties, grammatical variations, and inverse relations.

ACADEMIC RESTRICTION COMPLIANCE:
Strictly declarative and deterministic.
Explicitly defines logical properties (transitivity, symmetry, inversibility) without probabilistic modeling.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class RelationshipDefinition:
    """Represents a formal relational predicate."""
    canonical_predicate: str
    grammatical_variations: List[str] = field(default_factory=list)
    inverse_predicate: Optional[str] = None
    expected_subject_type: str = "Any"
    expected_object_type: str = "Any"
    is_transitive: bool = False
    is_symmetric: bool = False
    is_functional: bool = False  # e.g. capital_of(State -> exactly one Capital)
    supports_reverse: bool = True  # Can this relationship be queried in reverse
    question_type: str = "generic"  # Question type associated with this relationship
    description: str = ""


CANONICAL_RELATIONSHIPS: List[RelationshipDefinition] = [
    # 1. Identity & Name
    RelationshipDefinition(
        canonical_predicate="has name",
        grammatical_variations=["has name", "name is", "named", "called", "known as"],
        expected_subject_type="Person",
        expected_object_type="Concept",
        is_functional=True,
        supports_reverse=False,
        question_type="name",
        description="Associates an individual or speaker with their official name."
    ),

    # 2. Residence / Living
    RelationshipDefinition(
        canonical_predicate="lives in",
        grammatical_variations=["lives in", "live in", "resides in", "reside in", "staying in", "stays in", "stay in", "residing in"],
        expected_subject_type="Person",
        expected_object_type="Location",
        supports_reverse=True,
        question_type="residence",
        description="Captures the place of residence of a person."
    ),

    # 3. Employment / Work
    RelationshipDefinition(
        canonical_predicate="works at",
        grammatical_variations=["works at", "work at", "works for", "work for", "employed at", "employed by", "works in", "work in", "is employed at", "is employed by"],
        inverse_predicate="employs",
        expected_subject_type="Person",
        expected_object_type="Organization",
        supports_reverse=True,
        question_type="employment",
        description="Captures employment relationships between a person and company/employer."
    ),

    # 4. Education / Study
    RelationshipDefinition(
        canonical_predicate="studies at",
        grammatical_variations=["studies at", "study at", "student at", "enrolled at", "attends", "attend", "is enrolled at", "is studying at", "studies in", "study in"],
        expected_subject_type="Person",
        expected_object_type="Institution",
        supports_reverse=True,
        question_type="education",
        description="Captures educational institution affiliation."
    ),

    # 5. Role / Classification
    RelationshipDefinition(
        canonical_predicate="is a",
        grammatical_variations=["is a", "is an", "are a", "am a", "am an", "acts as", "works as"],
        expected_subject_type="Any",
        expected_object_type="Concept",
        supports_reverse=False,
        question_type="classification",
        description="Taxonomic or occupational classification."
    ),

    # 6. Capital
    RelationshipDefinition(
        canonical_predicate="capital of",
        grammatical_variations=["capital of", "is the capital of", "capital city of"],
        inverse_predicate="has capital",
        expected_subject_type="City",
        expected_object_type="Region",
        is_functional=True,
        supports_reverse=True,
        question_type="capital",
        description="Captures capital city administrative status."
    ),

    # 7. Geographic Location
    RelationshipDefinition(
        canonical_predicate="located in",
        grammatical_variations=["located in", "situated in", "is in", "lies in", "found in"],
        expected_subject_type="Location",
        expected_object_type="Location",
        is_transitive=True,
        supports_reverse=True,
        question_type="location",
        description="Geographic containment relation."
    ),

    # 8. Administrative Subdivisions
    RelationshipDefinition(
        canonical_predicate="is a state in",
        grammatical_variations=["is a state in", "state in"],
        expected_subject_type="State",
        expected_object_type="Country",
        supports_reverse=True,
        question_type="location",
        description="Administrative federal state subdivision."
    ),
    RelationshipDefinition(
        canonical_predicate="is a country in",
        grammatical_variations=["is a country in", "country in"],
        expected_subject_type="Country",
        expected_object_type="Continent",
        supports_reverse=True,
        question_type="location",
        description="National geographic subdivision within a continent."
    ),
    RelationshipDefinition(
        canonical_predicate="is a city in",
        grammatical_variations=["is a city in", "city in"],
        expected_subject_type="City",
        expected_object_type="Region",
        supports_reverse=True,
        question_type="location",
        description="Urban municipal location within a state or country."
    ),

    # 9. Part-Whole
    RelationshipDefinition(
        canonical_predicate="part of",
        grammatical_variations=["part of", "component of", "member of", "element of"],
        is_transitive=True,
        supports_reverse=True,
        question_type="part_whole",
        description="Meronymic part-whole relationship."
    ),

    # 10. Border / Adjacency
    RelationshipDefinition(
        canonical_predicate="borders",
        grammatical_variations=["borders", "border", "neighbors", "neighbor", "adjacent to"],
        is_symmetric=True,
        supports_reverse=True,
        question_type="border",
        description="Geographic adjacency relationship."
    ),

    # 11. Inception / Establishment
    RelationshipDefinition(
        canonical_predicate="established in",
        grammatical_variations=["established in", "founded in", "created in", "built in", "started in"],
        expected_subject_type="Organization",
        expected_object_type="Date",
        is_functional=True,
        supports_reverse=False,
        question_type="establishment",
        description="Inception date or founding year."
    ),

    # 12. Authorship / Creator
    RelationshipDefinition(
        canonical_predicate="developed by",
        grammatical_variations=["developed by", "invented by", "authored by", "written by", "created by", "discovered by"],
        inverse_predicate="developed",
        expected_subject_type="Product",
        expected_object_type="Person",
        supports_reverse=True,
        question_type="creator",
        description="Creator or discoverer relationship."
    ),

    # 13. Production / Manufacturing
    RelationshipDefinition(
        canonical_predicate="produces",
        grammatical_variations=["produces", "produce", "manufactures", "manufacture"],
        expected_subject_type="Entity",
        expected_object_type="Product",
        supports_reverse=True,
        question_type="production",
        description="Commercial or natural production of goods."
    ),

    # 14. Membership / Belongs To
    RelationshipDefinition(
        canonical_predicate="belongs to",
        grammatical_variations=["belongs to", "belong to", "affiliated with"],
        supports_reverse=True,
        question_type="location",
        description="Institutional or geopolitical belonging."
    ),

    # 15. Possession / Ownership
    RelationshipDefinition(
        canonical_predicate="has",
        grammatical_variations=["has", "have", "owns", "own", "possesses", "possess", "holds", "hold"],
        inverse_predicate="owned by",
        expected_subject_type="Entity",
        expected_object_type="Entity",
        is_transitive=True,
        supports_reverse=True,
        question_type="possession",
        description="Captures possession, ownership, or custody relationship."
    ),
]


class RelationshipRegistry:
    """
    Central deterministic registry for relational predicates and logical properties.
    """

    def __init__(self):
        self._predicates: Dict[str, RelationshipDefinition] = {}
        self._variation_to_canonical: Dict[str, str] = {}
        self._load_relationships()

    def _load_relationships(self):
        for rel in CANONICAL_RELATIONSHIPS:
            canon = rel.canonical_predicate.lower()
            self._predicates[canon] = rel
            self._variation_to_canonical[canon] = canon
            for var in rel.grammatical_variations:
                self._variation_to_canonical[var.lower()] = canon

    def get_canonical_predicate(self, predicate: str) -> str:
        """Map any grammatical variation to its canonical predicate."""
        if not predicate:
            return ""
        norm = predicate.strip().lower()
        return self._variation_to_canonical.get(norm, norm)

    def get_definition(self, predicate: str) -> Optional[RelationshipDefinition]:
        """Get the formal RelationshipDefinition for a given predicate."""
        canon = self.get_canonical_predicate(predicate)
        return self._predicates.get(canon)

    def is_transitive(self, predicate: str) -> bool:
        rel = self.get_definition(predicate)
        return rel.is_transitive if rel else False

    def is_symmetric(self, predicate: str) -> bool:
        rel = self.get_definition(predicate)
        return rel.is_symmetric if rel else False

    def get_inverse(self, predicate: str) -> Optional[str]:
        rel = self.get_definition(predicate)
        return rel.inverse_predicate if rel else None

    def get_all_variations(self, predicate: str) -> List[str]:
        rel = self.get_definition(predicate)
        if rel:
            return [rel.canonical_predicate] + rel.grammatical_variations
        return [predicate]

    def find_predicate_by_phrase(self, phrase: str) -> Optional[str]:
        """Map any verb/relational phrase variation to its canonical predicate."""
        if not phrase:
            return None
        norm = phrase.strip().lower()
        return self._variation_to_canonical.get(norm)

    def is_reverse_supported(self, predicate: str) -> bool:
        """Check whether reverse lookup is supported for the given predicate."""
        defn = self.get_definition(predicate)
        return defn.supports_reverse if defn else False

    def is_functional(self, predicate: str) -> bool:
        """Check whether the given predicate is functional (single-valued)."""
        defn = self.get_definition(predicate)
        return defn.is_functional if defn else False

    def get_question_type(self, predicate: str) -> str:
        """Get the question type associated with the given predicate."""
        defn = self.get_definition(predicate)
        return defn.question_type if defn else "generic"

    def match_reverse_phrase(self, text: str) -> Optional[Tuple[str, str, RelationshipDefinition]]:
        """
        Check if text begins with a registered relational phrase.
        Returns (matched_phrase, remaining_entity_text, RelationshipDefinition) or None.
        Matches longest variation first for deterministic ambiguity resolution.
        """
        if not text:
            return None
        norm_text = text.strip()
        lower_text = norm_text.lower()

        # Sort variations by length descending for greedy match
        for var, canon in sorted(self._variation_to_canonical.items(), key=lambda x: len(x[0]), reverse=True):
            if lower_text.startswith(var + " ") or lower_text == var:
                entity_part = norm_text[len(var):].strip()
                defn = self.get_definition(canon)
                if defn:
                    return var, entity_part, defn
        return None


GLOBAL_RELATIONSHIP_REGISTRY = RelationshipRegistry()
