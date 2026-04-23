import { useEffect, useRef } from "react";


interface Props {
  json: string;
  className?: string;
}

export default function PlotlyChart({ json, className }: Props) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;

    let fig: { data: unknown[]; layout: Record<string, unknown> };
    try {
      fig = JSON.parse(json);
    } catch {
      return;
    }

    import("plotly.js-dist-min").then((module) => {
      const P = module.default ?? module;
      P.newPlot(
        el,
        fig.data ?? [],
        {
          ...fig.layout,
          paper_bgcolor: "rgba(0,0,0,0)",
          plot_bgcolor: "rgba(0,0,0,0)",
          font: { color: "#e0e0e0", ...((fig.layout?.font as object) ?? {}) },
          autosize: true,
        },
        { displayModeBar: false, responsive: true }
      );
    });

    return () => {
      import("plotly.js-dist-min").then((module) => {
        const P = module.default ?? module;
        P.purge(el);
      });
    };
  }, [json]);

  return <div ref={ref} className={className} style={{ width: "100%", height: "100%" }} />;
}
