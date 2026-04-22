import Plot from "react-plotly.js";

interface Props {
  json: string;
  className?: string;
}

export default function PlotlyChart({ json, className }: Props) {
  const fig = JSON.parse(json) as { data: Plotly.Data[]; layout: Partial<Plotly.Layout> };
  return (
    <div className={className}>
      <Plot
        data={fig.data}
        layout={{
          ...fig.layout,
          paper_bgcolor: "rgba(0,0,0,0)",
          plot_bgcolor: "rgba(0,0,0,0)",
          font: { color: "#e0e0e0", ...(fig.layout.font ?? {}) },
        }}
        config={{ displayModeBar: false, responsive: true }}
        style={{ width: "100%", height: "100%" }}
        useResizeHandler
      />
    </div>
  );
}
