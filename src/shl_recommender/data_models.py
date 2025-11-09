"""Shared data models for the SHL recommender."""

from __future__ import annotations

import re
from typing import List, Optional, Set

from pydantic import BaseModel, Field, HttpUrl


ASSESSMENT_TYPE_LABELS = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behaviour",
    "S": "Simulations",
}


def _bool_to_label(value: Optional[bool]) -> str:
    if value is True:
        return "Yes"
    if value is False:
        return "No"
    return "Unknown"


class AssessmentMetadata(BaseModel):
    """Core metadata captured for each catalog assessment."""

    entity_id: str
    name: str
    url: HttpUrl
    assessment_types: Set[str] = Field(default_factory=set)
    description: Optional[str] = None
    job_levels: List[str] = Field(default_factory=list)
    languages: List[str] = Field(default_factory=list)
    assessment_length: Optional[str] = None
    remote_testing: Optional[bool] = None
    adaptive: Optional[bool] = None

    def combined_text(self) -> str:
        """Build a single text blob for embedding."""

        sections = [self.name]
        if self.description:
            sections.append(self.description)
        if self.job_levels:
            sections.append("Job Levels: " + ", ".join(self.job_levels))
        if self.languages:
            sections.append("Languages: " + ", ".join(self.languages))
        if self.assessment_length:
            sections.append(f"Assessment Length: {self.assessment_length}")
        if self.assessment_types:
            sections.append("Types: " + ", ".join(sorted(self.assessment_types)))
        return "\n".join(sections)

    def duration_minutes(self) -> Optional[int]:
        """Extract the numeric duration (in minutes) when present."""

        if not self.assessment_length:
            return None
        match = re.findall(r"\d+", self.assessment_length)
        if not match:
            return None
        try:
            return int(match[0])
        except ValueError:
            return None

    def human_readable_types(self) -> List[str]:
        """Return descriptive labels for assessment type codes."""

        labels: List[str] = []
        for code in sorted(self.assessment_types):
            normalized = code.strip().upper()
            labels.append(ASSESSMENT_TYPE_LABELS.get(normalized, normalized))
        return labels

    def remote_label(self) -> str:
        return _bool_to_label(self.remote_testing)

    def adaptive_label(self) -> str:
        return _bool_to_label(self.adaptive)

    def to_recommendation_item(self) -> "RecommendationItem":
        return RecommendationItem(
            name=self.name,
            url=self.url,
            description=self.description,
            duration=self.duration_minutes(),
            remote_support=self.remote_label(),
            adaptive_support=self.adaptive_label(),
            test_type=self.human_readable_types(),
        )


class RecommendationRequest(BaseModel):
    """Request payload for the recommendation API."""

    query: str


class RecommendationItem(BaseModel):
    """Individual recommendation item returned to clients."""

    name: str
    url: HttpUrl
    description: Optional[str] = None
    duration: Optional[int] = None
    remote_support: str = "Unknown"
    adaptive_support: str = "Unknown"
    test_type: List[str] = Field(default_factory=list)


class RecommendationResponse(BaseModel):
    """Response payload for the recommendation API."""

    recommended_assessments: List[RecommendationItem] = Field(default_factory=list)


