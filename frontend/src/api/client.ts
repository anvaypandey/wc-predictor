export interface TeamStats {
  fifa_rank: string;
  games: number;
  win_rate: string;
  draw_rate: string;
  avg_scored: string;
  avg_conceded: string;
  recent_form: string;
}

export interface H2HRecord {
  home_wins: number;
  draws: number;
  away_wins: number;
  total: number;
}

export interface PredictResponse {
  home: string;
  away: string;
  predicted: string;
  win_prob: number;
  draw_prob: number;
  loss_prob: number;
  home_stats: TeamStats;
  away_stats: TeamStats;
  h2h: H2HRecord;
  prob_chart: string;
  shap_chart: string;
}

export interface TeamsResponse {
  teams: string[];
  rankings: Record<string, { rank: number }>;
  groups: Record<string, string[]>;
}

export interface WCMatch {
  date: string;
  home_team: string;
  away_team: string;
  score: string;
  actual: string;
  predicted: string;
  correct: boolean;
}

export interface AccuracyResponse {
  cv_accuracy: number;
  cv_std: number;
  n_training_rows: number;
  model_type: string;
  wc_accuracy: number;
  wc_matches: number;
  wc_years: number[];
  backtest_by_year: Record<number, WCMatch[]>;
  feature_chart: string;
  confusion_chart: string;
  wc_year_chart: string;
  calibration_chart: string;
  tournament_chart: string;
}

export interface SimProgress {
  type: "progress";
  stage: "groups" | "knockout";
  pct: number;
  label: string;
}

export interface SimResult {
  type: "result";
  data: {
    group_results: Record<string, Record<string, number>[]>;
    likely_standings: Record<string, string[]>;
    bracket: string[];
    ko_results: Record<string, Record<string, number>>;
    bracket_chart: string;
    win_chart: string;
  };
}

export async function fetchTeams(): Promise<TeamsResponse> {
  const res = await fetch("/api/teams");
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function fetchPrediction(
  home: string,
  away: string,
  neutral: boolean
): Promise<PredictResponse> {
  const res = await fetch("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ home, away, neutral }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export function streamSimulation(
  nSims: number,
  onProgress: (e: SimProgress) => void,
  onResult: (e: SimResult) => void,
  onError: (err: Event) => void
): () => void {
  const es = new EventSource(`/api/simulate/stream?n_sims=${nSims}`);
  es.onmessage = (e) => {
    const msg = JSON.parse(e.data) as SimProgress | SimResult;
    if (msg.type === "progress") onProgress(msg);
    else if (msg.type === "result") { onResult(msg); es.close(); }
  };
  es.onerror = (e) => { onError(e); es.close(); };
  return () => es.close();
}

export async function fetchAccuracy(): Promise<AccuracyResponse> {
  const res = await fetch("/api/accuracy");
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}
