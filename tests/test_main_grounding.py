import pytest

from app.main_grounding import verify_direct_sql_grounding
from app.schemas import PlanGroundingError
from scripts.query_context import EntityLink, QueryLinks


def links():
    return QueryLinks(players=[EntityLink(alias="库里", id="201939", display_name="Stephen Curry")],
                      seasons=["2023-24"], phases=["regular_season"])


SCOPE = " FROM player_season_stat WHERE person_id = '201939' AND season_id = '2023-24' AND competition_phase = 'regular_season'"


@pytest.mark.parametrize("projection,question", [
    ("points_total", "库里 2023-24 常规赛 PPG"),
    ("points_total", "库里 2023-24 常规赛 assists"),
    ("points_total", "库里 2023-24 常规赛 rebounds"),
    ("points_total", "库里 2023-24 常规赛 eFG%"),
    ("points_total", "库里 2023-24 常规赛总得分和总助攻"),
    ("rebounds_total AS points_total", "库里 2023-24 常规赛总得分"),
])
def test_direct_sql_requires_every_requested_metric(projection, question):
    with pytest.raises(PlanGroundingError, match="metric"):
        verify_direct_sql_grounding("SELECT " + projection + SCOPE, question, links())


@pytest.mark.parametrize("suffix", [" OR 1=1", " OR person_id = '2544'"])
def test_direct_sql_rejects_scope_bypass(suffix):
    with pytest.raises(PlanGroundingError):
        verify_direct_sql_grounding("SELECT points_total" + SCOPE + suffix, "库里 2023-24 常规赛总得分", links())


def test_direct_sql_accepts_grounded_complete_projection():
    verify_direct_sql_grounding("SELECT points_total, assists_total" + SCOPE,
                                "库里 2023-24 常规赛总得分和总助攻", links())
