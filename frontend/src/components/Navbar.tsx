import { NavLink } from "react-router-dom";

const links = [
  { to: "/", label: "Match Predictor" },
  { to: "/simulate", label: "Bracket Simulator" },
  { to: "/accuracy", label: "Model Accuracy" },
];

export default function Navbar() {
  return (
    <nav className="bg-[#1a1d27] border-b border-[#2e303a] sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 flex items-center gap-8 h-14">
        <span className="text-[#f39c12] font-bold text-lg tracking-tight">
          ⚽ WC 2026
        </span>
        <div className="flex gap-1">
          {links.map(({ to, label }) => (
            <NavLink
              key={to}
              to={to}
              end={to === "/"}
              className={({ isActive }) =>
                `px-4 py-1.5 rounded text-sm font-medium transition-colors ${
                  isActive
                    ? "bg-[#f39c12] text-black"
                    : "text-[#aaa] hover:text-white hover:bg-[#2e303a]"
                }`
              }
            >
              {label}
            </NavLink>
          ))}
        </div>
      </div>
    </nav>
  );
}
