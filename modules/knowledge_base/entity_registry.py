"""
Entity Registry Module
Manages entity identification, canonical names, types, and alias resolution.

ACADEMIC RESTRICTION COMPLIANCE:
Strictly deterministic and dictionary/rule-based.
No named entity recognition (NER) models, no embeddings, no fuzzy string distance clustering.
All aliases and entity types are either explicitly asserted in documents or registered via canonical dictionaries.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
import re


@dataclass
class Entity:
    """Represents a canonical knowledge entity."""
    entity_id: str
    canonical_name: str
    entity_type: str = "Unknown"  # Person, Organization, Institution, City, State, Country, Location, Concept, Other
    aliases: Set[str] = field(default_factory=set)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'entity_id': self.entity_id,
            'canonical_name': self.canonical_name,
            'entity_type': self.entity_type,
            'aliases': sorted(list(self.aliases))
        }


# Canonical well-known entity types and geographic/institutional aliases
INITIAL_ENTITY_ALIASES: Dict[str, Tuple[str, str, List[str]]] = {
    # canonical_name: (entity_id, entity_type, [aliases])
    "Bangalore": ("city:bangalore", "City", ["Bengaluru", "Bangalore City", "Bengaluru City"]),
    "Bengaluru": ("city:bangalore", "City", ["Bangalore"]),
    "Karnataka": ("state:karnataka", "State", []),
    "India": ("country:india", "Country", ["Republic of India", "Bharat"]),
    "United States": ("country:usa", "Country", ["USA", "United States of America", "U.S.", "U.S.A."]),
    "United Kingdom": ("country:uk", "Country", ["UK", "U.K.", "Britain", "Great Britain"]),
    "RUAS": ("inst:ruas", "Institution", ["M S Ramaiah University of Applied Sciences", "Ramaiah University", "MSRUAS"]),
    "KPMG": ("org:kpmg", "Organization", []),
    "Asia": ("loc:asia", "Continent", []),
    "Europe": ("loc:europe", "Continent", []),
    "Python": ("concept:python", "ProgrammingLanguage", ["Python Language"]),
}


class EntityRegistry:
    """
    Central deterministic registry for entity tracking and alias resolution.
    """

    def __init__(self):
        self._entities: Dict[str, Entity] = {}  # entity_id -> Entity
        self._name_to_id: Dict[str, str] = {}  # normalized_name -> entity_id
        self._load_initial_data()

    def _load_initial_data(self):
        """Seed registry with standard canonical entities and aliases."""
        for canon_name, (eid, etype, aliases) in INITIAL_ENTITY_ALIASES.items():
            self.register_entity(canonical_name=canon_name, entity_type=etype, entity_id=eid, aliases=aliases)

    @staticmethod
    def normalize_name(name: str) -> str:
        """Normalize an entity string for deterministic comparison."""
        if not name:
            return ""
        s = name.strip()
        # Collapse multiple spaces
        s = re.sub(r'\s+', ' ', s)
        # Strip leading articles
        s = re.sub(r'^(?:the|a|an)\s+', '', s, flags=re.IGNORECASE).strip()
        return s.lower()

    def register_entity(self, canonical_name: str, entity_type: str = "Unknown",
                        entity_id: Optional[str] = None, aliases: Optional[List[str]] = None) -> Entity:
        """Register or update an entity in the registry."""
        norm_canon = self.normalize_name(canonical_name)
        if not norm_canon:
            raise ValueError("Entity canonical name cannot be empty.")

        # Check if already registered
        if norm_canon in self._name_to_id:
            existing_id = self._name_to_id[norm_canon]
            entity = self._entities[existing_id]
            if entity_type != "Unknown" and entity.entity_type == "Unknown":
                entity.entity_type = entity_type
            if aliases:
                for a in aliases:
                    entity.aliases.add(a)
                    self._name_to_id[self.normalize_name(a)] = existing_id
            return entity

        eid = entity_id or f"ent:{re.sub(r'[^a-z0-9_]+', '_', norm_canon)}"
        # Ensure unique ID
        base_eid = eid
        counter = 1
        while eid in self._entities and self._entities[eid].canonical_name.lower() != canonical_name.lower():
            eid = f"{base_eid}_{counter}"
            counter += 1

        entity = Entity(
            entity_id=eid,
            canonical_name=canonical_name.strip(),
            entity_type=entity_type,
            aliases=set(aliases or [])
        )
        self._entities[eid] = entity
        self._name_to_id[norm_canon] = eid

        if aliases:
            for a in aliases:
                norm_a = self.normalize_name(a)
                if norm_a:
                    self._name_to_id[norm_a] = eid

        return entity

    def resolve_entity(self, name: str) -> Optional[Entity]:
        """
        Resolve an entity mention to its canonical Entity object.
        Returns None if not found in registry.
        """
        if not name:
            return None
        norm = self.normalize_name(name)
        eid = self._name_to_id.get(norm)
        if eid and eid in self._entities:
            return self._entities[eid]
        return None

    def get_canonical_name(self, name: str) -> str:
        """Return the canonical name of an entity if known, else return original stripped name."""
        entity = self.resolve_entity(name)
        if entity:
            return entity.canonical_name
        return name.strip()

    def are_aliases(self, name1: str, name2: str) -> bool:
        """Check if two entity mentions refer to the same canonical entity."""
        if not name1 or not name2:
            return False
        if self.normalize_name(name1) == self.normalize_name(name2):
            return True
        e1 = self.resolve_entity(name1)
        e2 = self.resolve_entity(name2)
        if e1 and e2 and e1.entity_id == e2.entity_id:
            return True
        return False

    def get_all_aliases(self, name: str) -> Set[str]:
        """Return all known names/aliases for an entity including the canonical name."""
        entity = self.resolve_entity(name)
        if entity:
            res = {entity.canonical_name}
            res.update(entity.aliases)
            return res
        return {name.strip()}


# Module-level singleton
GLOBAL_ENTITY_REGISTRY = EntityRegistry()
