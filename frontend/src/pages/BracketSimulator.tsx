import { useEffect, useRef, useState } from "react";
import PlotlyChart from "../components/PlotlyChart";
import { fetchTeams, streamSimulation } from "../api/client";
import type { SimResult, TeamsResponse, TeamGroupStats } from "../api/client";

type Stage = "idle" | "groups" | "knockout" | "done" | "error";

export default function BracketSimulator() {
  const [teamsData, setTeamsData] = useState<TeamsResponse | null>(null);
  const [nSims, setNSims] = useState(500);
  const [stage, setStage] = useState<Stage>("idle");
  const [pct, setPct] = useState(0);
  const [label, setLabel] = useState("");
  const [result, setResult] = useState<SimResult["data"] | null>(null);
  const [error, setError] = useState("");
  const cleanupRef = useRef<(() => void) | null>(null);

  useEffect(() => {
    fetchTeams().then(setTeamsData).catch((e) => setError(e.message));
    return () => cleanupRef.current?.();
  }, []);

  function runSim() {
    setStage("groups");
    setPct(0);
    setLabel("Starting…");
    setResult(null);
    setError("");

    const close = streamSimulation(
      nSims,
      (ev) => {
        setStage(ev.stage as Stage);
        setPct(ev.pct);
        setLabel(ev.label);
      },
      (ev) => {
        setResult(ev.data);
        setStage("done");
      },
      () => {
        setError("Simulation failed. Is the backend running?");
        setStage("error");
      }
    );
    cleanupRef.current = close;
  }

  const groups = teamsData?.groups ?? {};
  const groupNames = Object.keys(groups).sort();
  const running = stage === "groups" || stage === "knockout";

  return (
    <div className="max-w-6xl mx-auto px-4 py-8 space-y-8">
      <h1 className="text-3xl font-bold text-white">Bracket Simulator</h1>

      {/* Groups grid */}
      {groupNames.length > 0 && (
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-4">
          {groupNames.map((g) => (
            <div key={g} className="bg-[#1a1d27] border border-[#2e303a] rounded-xl p-4">
              <h3 className="text-[#f39c12] font-semibold text-sm mb-2">Group {g}</h3>
              <ul className="space-y-1">
                {groups[g].map((team) => (
                  <li key={team} className="text-sm text-[#ccc]">{team}</li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      )}

      {/* Controls */}
      <div className="bg-[#1a1d27] border border-[#2e303a] rounded-xl p-6 space-y-5">
        <div className="space-y-2">
          <label className="text-xs uppercase tracking-widest text-[#aaa]">
            Simulations: <span className="text-white font-semibold">{nSims.toLocaleString()}</span>
          </label>
          <input
            type="range" min={100} max={2000} step={100}
            value={nSims}
            onChange={(e) => setNSims(Number(e.target.value))}
            className="w-full accent-[#f39c12]"
          />
          <div className="flex justify-between text-xs text-[#666]">
            <span>100</span><span>2,000</span>
          </div>
        </div>
        <button
          className="px-6 py-2.5 bg-[#f39c12] text-black font-semibold rounded-lg hover:bg-[#e67e22] transition-colors disabled:opacity-50"
          onClick={runSim}
          disabled={running}
        >
          {running ? "Simulating…" : "Run Simulation"}
        </button>
        {error && <p className="text-red-400 text-sm">{error}</p>}

        {/* Progress */}
        {running && (
          <div className="space-y-2">
            <div className="flex justify-between text-sm text-[#aaa]">
              <span>{label}</span>
              <span>{pct}%</span>
            </div>
            <div className="w-full bg-[#2e303a] rounded-full h-2">
              <div
                className="bg-[#f39c12] h-2 rounded-full transition-all duration-300"
                style={{ width: `${pct}%` }}
              />
            </div>
          </div>
        )}
      </div>

      {/* Results */}
      {result && (
        <div className="space-y-8">
          {/* Group standings */}
          <div>
            <h2 className="text-xl font-bold text-white mb-4">Most Likely Group Standings</h2>
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-4">
              {groupNames.map((g) => {
                const standings = result.likely_standings[g] ?? [];
                return (
                  <div key={g} className="bg-[#1a1d27] border border-[#2e303a] rounded-xl p-4">
                    <h3 className="text-[#f39c12] font-semibold text-sm mb-2">Group {g}</h3>
                    <ol className="space-y-1">
                      {standings.map((team, i) => {
                        const stats: TeamGroupStats | undefined = result.group_results[team];
                        const advPct = stats ? stats.advance_pct.toFixed(0) : null;
                        return (
                          <li key={team} className="flex items-center gap-2 text-sm">
                            <span className={`w-5 text-center font-bold ${i < 2 ? "text-[#f39c12]" : "text-[#666]"}`}>{i + 1}</span>
                            <span className="text-[#ccc] flex-1">{team}</span>
                            {advPct && <span className="text-xs text-[#888]">{advPct}%</span>}
                          </li>
                        );
                      })}
                    </ol>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Win probability chart */}
          <div>
            <h2 className="text-xl font-bold text-white mb-2">Championship Probability</h2>
            <PlotlyChart json={result.win_chart} className="h-[440px]" />
          </div>

          {/* Bracket */}
          <div>
            <h2 className="text-xl font-bold text-white mb-2">Knockout Bracket</h2>
            <PlotlyChart json={result.bracket_chart} className="h-[620px]" />
          </div>

          {/* Round-by-round table */}
          <div>
            <h2 className="text-xl font-bold text-white mb-4">Round-by-Round Probabilities</h2>
            <div className="overflow-x-auto">
              <table className="w-full text-sm border-collapse">
                <thead>
                  <tr className="border-b border-[#2e303a]">
                    {["Team", "R16", "QF", "SF", "Final", "Winner"].map((h) => (
                      <th key={h} className="px-3 py-2 text-left text-[#888] font-medium">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {result.bracket
                    .slice()
                    .sort((a, b) => (result.ko_results[b]?.Winner ?? 0) - (result.ko_results[a]?.Winner ?? 0))
                    .map((team) => {
                      const r = result.ko_results[team] ?? {};
                      return (
                        <tr key={team} className="border-b border-[#1f2028] hover:bg-[#1a1d27]">
                          <td className="px-3 py-2 text-white font-medium">{team}</td>
                          {["R16", "QF", "SF", "Final", "Winner"].map((rnd) => (
                            <td key={rnd} className="px-3 py-2 text-[#ccc]">
                              {r[rnd] != null ? `${r[rnd].toFixed(1)}%` : "—"}
                            </td>
                          ))}
                        </tr>
                      );
                    })}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
