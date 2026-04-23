import { useEffect, useState } from "react";
import PlotlyChart from "../components/PlotlyChart";
import StatCard from "../components/StatCard";
import { fetchAccuracy } from "../api/client";
import type { AccuracyResponse, WCMatch } from "../api/client";

export default function ModelAccuracy() {
  const [data, setData] = useState<AccuracyResponse | null>(null);
  const [selectedYear, setSelectedYear] = useState<number | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    fetchAccuracy()
      .then((d) => {
        setData(d);
        if (d.wc_years.length > 0) setSelectedYear(d.wc_years[0]);
      })
      .catch((e) => setError(e.message));
  }, []);

  if (error) return (
    <div className="max-w-5xl mx-auto px-4 py-8">
      <p className="text-red-400">{error}</p>
    </div>
  );

  if (!data) return (
    <div className="max-w-5xl mx-auto px-4 py-8 text-[#888]">Loading…</div>
  );

  const matches: WCMatch[] = selectedYear != null ? (data.backtest_by_year[String(selectedYear)] ?? []) : [];

  return (
    <div className="max-w-6xl mx-auto px-4 py-8 space-y-8">
      <h1 className="text-3xl font-bold text-white">Model Accuracy</h1>

      {/* Metric cards */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <StatCard
          label="CV Accuracy"
          value={`${data.cv_accuracy}%`}
          sub={`± ${data.cv_std}%`}
        />
        <StatCard
          label="WC Accuracy"
          value={`${data.wc_accuracy}%`}
          sub={`${data.wc_matches} matches`}
        />
        <StatCard
          label="Training rows"
          value={data.n_training_rows.toLocaleString()}
        />
        <StatCard
          label="Model"
          value={data.model_type}
        />
      </div>

      {/* Feature importance */}
      <div>
        <h2 className="text-xl font-bold text-white mb-2">Feature Importance</h2>
        <PlotlyChart json={data.feature_chart} className="h-[540px]" />
      </div>

      {/* WC by year + confusion side by side */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div>
          <h2 className="text-xl font-bold text-white mb-2">WC Accuracy by Year</h2>
          <PlotlyChart json={data.wc_year_chart} className="h-[300px]" />
        </div>
        <div>
          <h2 className="text-xl font-bold text-white mb-2">Confusion Matrix</h2>
          <PlotlyChart json={data.confusion_chart} className="h-[300px]" />
        </div>
      </div>

      {/* Per-year match table */}
      <div>
        <div className="flex items-center gap-4 mb-4">
          <h2 className="text-xl font-bold text-white">World Cup Backtest</h2>
          <select
            className="bg-[#1a1d27] border border-[#2e303a] rounded-lg px-3 py-1.5 text-white text-sm focus:outline-none focus:border-[#f39c12]"
            value={selectedYear ?? ""}
            onChange={(e) => setSelectedYear(Number(e.target.value))}
          >
            {data.wc_years.map((y) => (
              <option key={y} value={y}>{y}</option>
            ))}
          </select>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm border-collapse">
            <thead>
              <tr className="border-b border-[#2e303a]">
                {["Date", "Home", "Away", "Score", "Actual", "Predicted", ""].map((h, i) => (
                  <th key={i} className="px-3 py-2 text-left text-[#888] font-medium">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {matches.map((m, i) => (
                <tr key={i} className="border-b border-[#1f2028] hover:bg-[#1a1d27]">
                  <td className="px-3 py-2 text-[#888]">{m.date}</td>
                  <td className="px-3 py-2 text-white">{m.home_team}</td>
                  <td className="px-3 py-2 text-white">{m.away_team}</td>
                  <td className="px-3 py-2 text-[#ccc]">{m.score}</td>
                  <td className="px-3 py-2 text-[#ccc]">{m.actual}</td>
                  <td className="px-3 py-2 text-[#ccc]">{m.predicted}</td>
                  <td className="px-3 py-2">
                    <span className={`text-xs font-bold px-2 py-0.5 rounded-full ${m.correct ? "bg-green-900 text-green-300" : "bg-red-900 text-red-300"}`}>
                      {m.correct ? "✓" : "✗"}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Calibration */}
      <div>
        <h2 className="text-xl font-bold text-white mb-2">Calibration Curve</h2>
        <PlotlyChart json={data.calibration_chart} className="h-[340px]" />
      </div>

      {/* Tournament accuracy */}
      <div>
        <h2 className="text-xl font-bold text-white mb-2">Accuracy by Tournament</h2>
        <PlotlyChart json={data.tournament_chart} className="h-[500px]" />
      </div>
    </div>
  );
}
