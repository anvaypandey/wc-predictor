from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    home:    str  = Field(..., min_length=1)
    away:    str  = Field(..., min_length=1)
    neutral: bool = True


class TeamStats(BaseModel):
    fifa_rank:      str
    games:          int
    win_rate:       str
    draw_rate:      str
    avg_scored:     str
    avg_conceded:   str
    recent_form:    str


class H2HRecord(BaseModel):
    home_wins:  int
    draws:      int
    away_wins:  int
    total:      int


class PredictResponse(BaseModel):
    home:             str
    away:             str
    predicted:        str          # "Win" | "Draw" | "Loss"
    win_prob:         float
    draw_prob:        float
    loss_prob:        float
    home_stats:       TeamStats
    away_stats:       TeamStats
    h2h:              H2HRecord
    prob_chart:       str          # Plotly figure JSON
    shap_chart:       str          # Plotly figure JSON


class TeamsResponse(BaseModel):
    teams:    list[str]
    rankings: dict[str, dict]
    groups:   dict[str, list[str]]


class WCMatch(BaseModel):
    date:      str
    home_team: str
    away_team: str
    score:     str
    actual:    str
    predicted: str
    correct:   bool


class AccuracyResponse(BaseModel):
    cv_accuracy:        float
    cv_std:             float
    n_training_rows:    int
    model_type:         str
    wc_accuracy:        float
    wc_matches:         int
    wc_years:           list[int]
    backtest_by_year:   dict[int, list[WCMatch]]
    feature_chart:      str   # Plotly JSON
    confusion_chart:    str   # Plotly JSON
    wc_year_chart:      str   # Plotly JSON
    calibration_chart:  str   # Plotly JSON
    tournament_chart:   str   # Plotly JSON
