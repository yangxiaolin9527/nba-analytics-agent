"""Language requirements, independent of the production capability registry."""
from dataclasses import dataclass
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field

class Strict(BaseModel):
    model_config=ConfigDict(extra='forbid',strict=True)

class Reference(Strict):
    mention: str=Field(min_length=1,max_length=120)
    canonical_name: str|None=None

class Predicate(Strict):
    field: Literal['points','assists','rebounds_total']
    operator: Literal['ge','gt','le','lt','eq']
    value: int=Field(ge=0,le=500)
    quote: str=Field(min_length=1)

class AnalysisRequest(Strict):
    operation: Literal['summary','window','efg','rank','game_count','clarify','unsupported']
    subject: Literal['player','team']='player'
    entity: Reference|None=None
    season_id: str|None=None
    competition_phase: Literal['regular_season','playoffs']|None=None
    metrics: list[str]=Field(default_factory=list,max_length=5)
    opponent: Reference|None=None
    date_from: str|None=None
    date_to: str|None=None
    last_n: int|None=Field(default=None,ge=1,le=100)
    sort_direction: Literal['asc','desc']='desc'
    limit: int|None=Field(default=None,ge=1,le=20)
    predicates: list[Predicate]=Field(default_factory=list,max_length=2)
    evidence: dict[str,str]=Field(default_factory=dict)
    reason: str|None=None

@dataclass(frozen=True)
class GroundedRequest:
    operation: str
    subject: str
    entity_id: str|None
    season_id: str
    competition_phase: str
    metrics: tuple[str,...]
    opponent_id: str|None=None
    date_from: str|None=None
    date_to: str|None=None
    last_n: int|None=None
    sort_direction: str='desc'
    limit: int|None=None
    predicates: tuple[tuple[str,str,int],...]=()
