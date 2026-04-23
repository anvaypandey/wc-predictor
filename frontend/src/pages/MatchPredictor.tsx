import { useEffect, useState, useCallback } from "react";
import { useSearchParams } from "react-router-dom";
import PlotlyChart from "../components/PlotlyChart";
import { fetchTeams, fetchPrediction } from "../api/client";
import type { PredictResponse, TeamsResponse } from "../api/client";

const OUTCOME_COLORS: Record<string, string> = {
  Win: "bg-green-700 text-green-100",
  Draw: "bg-yellow-700 text-yellow-100",
  Loss: "bg-red-700 text-red-100",
};

function confidenceLabel(win: number, draw: number, loss: number): { label: string; color: string } {
  const probs = [win, draw, loss].sort((a, b) => b - a);
  const margin = probs[0] - probs[1];
  if (margin >= 0.30) return { label: "High confidence", color: "text-green-400" };
  if (margin >= 0.15) return { label: "Medium confidence", color: "text-yellow-400" };
  return { label: "Low confidence", color: "text-orange-400" };
}

export default function MatchPredictor() {
  const [searchParams, setSearchParams] = useSearchParams();
  const [teamsData, setTeamsData] = useState<TeamsResponse | null>(null);
  const [home, setHome] = useState(searchParams.get("home") ?? "");
  const [away, setAway] = useState(searchParams.get("away") ?? "");
  const [neutral, setNeutral] = useState(searchParams.get("neutral") !== "false");
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [copied, setCopied] = useState(false);

  const runPrediction = useCallback(async (h: string, a: string, n: boolean) => {
    if (!h || !a || h === a) return;
    setLoading(true);
    setError("");
    try {
      const r = await fetchPrediction(h, a, n);
      setResult(r);
    } catch (e: unknown) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchTeams()
      .then((d) => {
        setTeamsData(d);
        const paramHome = searchParams.get("home");
        const paramAway = searchParams.get("away");
        if (paramHome && paramAway && paramHome !== paramAway) {
          const n = searchParams.get("neutral") !== "false";
          setHome(paramHome);
          setAway(paramAway);
          setNeutral(n);
          runPrediction(paramHome, paramAway, n);
        } else if (d.teams.length >= 2) {
          setHome((prev) => prev || d.teams[0]);
          setAway((prev) => prev || d.teams[1]);
        }
      })
      .catch((e) => setError(e.message));
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function handlePredict() {
    if (!home || !away || home === away) {
      setError("Select two different teams.");
      return;
    }
    setSearchParams({ home, away, neutral: String(neutral) });
    runPrediction(home, away, neutral);
  }

  function handleCopyLink() {
    const url = new URL(window.location.href);
    url.searchParams.set("home", home);
    url.searchParams.set("away", away);
    url.searchParams.set("neutral", String(neutral));
    navigator.clipboard.writeText(url.toString());
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }

  const teams = teamsData?.teams ?? [];

  return (
    <div className="max-w-5xl mx-auto px-4 py-8 space-y-8">
      <h1 className="text-3xl font-bold text-white">Match Predictor</h1>

      {/* Controls */}
      <div className="bg-[#1a1d27] border border-[#2e303a] rounded-xl p-6 space-y-5">
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div className="space-y-1">
            <label className="text-xs uppercase tracking-widest text-[#aaa]">Home Team</label>
            <select
              className="w-full bg-[#0f1117] border border-[#2e303a] rounded-lg px-3 py-2 text-white focus:outline-none focus:border-[#f39c12]"
              value={home}
              onChange={(e) => setHome(e.target.value)}
            >
              {teams.map((t) => <option key={t} value={t}>{t}</option>)}
            </select>
          </div>
          <div className="space-y-1">
            <label className="text-xs uppercase tracking-widest text-[#aaa]">Away Team</label>
            <select
              className="w-full bg-[#0f1117] border border-[#2e303a] rounded-lg px-3 py-2 text-white focus:outline-none focus:border-[#f39c12]"
              value={away}
              onChange={(e) => setAway(e.target.value)}
            >
              {teams.map((t) => <option key={t} value={t}>{t}</option>)}
            </select>
          </div>
        </div>
        <label className="flex items-center gap-3 cursor-pointer select-none">
          <input
            type="checkbox"
            className="w-4 h-4 accent-[#f39c12]"
            checked={neutral}
            onChange={(e) => setNeutral(e.target.checked)}
          />
          <span className="text-[#ccc] text-sm">Neutral venue</span>
        </label>
        <div className="flex items-center gap-3">
          <button
            className="px-6 py-2.5 bg-[#f39c12] text-black font-semibold rounded-lg hover:bg-[#e67e22] transition-colors disabled:opacity-50"
            onClick={handlePredict}
            disabled={loading || teams.length === 0}
          >
            {loading ? "Predicting…" : "Predict"}
          </button>
          {result && (
            <button
              className="px-4 py-2.5 bg-[#1a1d27] border border-[#2e303a] text-[#ccc] text-sm rounded-lg hover:border-[#f39c12] hover:text-white transition-colors"
              onClick={handleCopyLink}
            >
              {copied ? "Copied!" : "Copy link"}
            </button>
          )}
        </div>
        {error && <p className="text-red-400 text-sm">{error}</p>}
      </div>

      {/* Result */}
      {result && (
        <div className="space-y-6">
          {/* Outcome badge + confidence */}
          <div className="flex items-center gap-3 flex-wrap">
            <span className="text-xl font-semibold text-white">
              {result.home} vs {result.away}
            </span>
            <span className={`px-3 py-1 rounded-full text-sm font-bold ${OUTCOME_COLORS[result.predicted]}`}>
              {result.predicted === "Win" ? `${result.home} Win` :
               result.predicted === "Loss" ? `${result.away} Win` : "Draw"}
            </span>
            {(() => {
              const c = confidenceLabel(result.win_prob, result.draw_prob, result.loss_prob);
              return <span className={`text-sm font-medium ${c.color}`}>{c.label}</span>;
            })()}
          </div>

          {/* Prob chart */}
          <PlotlyChart json={result.prob_chart} className="h-[220px]" />

          {/* Team stats */}
          <div className="grid grid-cols-2 gap-6">
            {[
              { team: result.home, stats: result.home_stats },
              { team: result.away, stats: result.away_stats },
            ].map(({ team, stats }) => (
              <div key={team} className="bg-[#1a1d27] border border-[#2e303a] rounded-xl p-5">
                <h3 className="font-semibold text-white mb-3">{team}</h3>
                <table className="w-full text-sm text-[#ccc]">
                  <tbody>
                    {[
                      ["FIFA Rank", stats.fifa_rank],
                      ["Games", stats.games],
                      ["Win rate", stats.win_rate],
                      ["Draw rate", stats.draw_rate],
                      ["Avg scored", stats.avg_scored],
                      ["Avg conceded", stats.avg_conceded],
                      ["Recent form", stats.recent_form],
                    ].map(([k, v]) => (
                      <tr key={k} className="border-b border-[#2e303a] last:border-0">
                        <td className="py-1 text-[#888]">{k}</td>
                        <td className="py-1 text-right font-medium text-white">{v}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ))}
          </div>

          {/* H2H */}
          {result.h2h.total > 0 && (
            <div className="bg-[#1a1d27] border border-[#2e303a] rounded-xl p-5">
              <h3 className="font-semibold text-white mb-3">Head-to-Head ({result.h2h.total} games)</h3>
              <div className="grid grid-cols-3 text-center text-sm">
                <div>
                  <div className="text-2xl font-bold text-green-400">{result.h2h.home_wins}</div>
                  <div className="text-[#888]">{result.home} wins</div>
                </div>
                <div>
                  <div className="text-2xl font-bold text-yellow-400">{result.h2h.draws}</div>
                  <div className="text-[#888]">Draws</div>
                </div>
                <div>
                  <div className="text-2xl font-bold text-red-400">{result.h2h.away_wins}</div>
                  <div className="text-[#888]">{result.away} wins</div>
                </div>
              </div>
            </div>
          )}

          {/* SHAP chart */}
          <div>
            <h3 className="font-semibold text-white mb-2">Feature contributions (SHAP)</h3>
            <PlotlyChart json={result.shap_chart} className="h-[480px]" />
          </div>
        </div>
      )}
    </div>
  );
}
