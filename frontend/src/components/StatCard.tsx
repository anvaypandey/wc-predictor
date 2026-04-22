interface Props {
  label: string;
  value: string | number;
  sub?: string;
}

export default function StatCard({ label, value, sub }: Props) {
  return (
    <div className="bg-[#1a1d27] border border-[#2e303a] rounded-xl p-5 flex flex-col gap-1">
      <span className="text-[#aaa] text-xs uppercase tracking-widest">{label}</span>
      <span className="text-2xl font-bold text-white">{value}</span>
      {sub && <span className="text-[#888] text-sm">{sub}</span>}
    </div>
  );
}
