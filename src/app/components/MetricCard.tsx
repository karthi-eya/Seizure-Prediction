import { Activity, Brain, Heart, Zap } from "lucide-react";

interface MetricCardProps {
  title: string;
  value: string;
  unit: string;
  status: "normal" | "elevated" | "low";
  icon: "brain" | "heart" | "activity" | "zap";
}

export function MetricCard({ title, value, unit, status, icon }: MetricCardProps) {
  const statusColors = {
    normal: "text-blue-600 bg-blue-50",
    elevated: "text-amber-600 bg-amber-50",
    low: "text-gray-600 bg-gray-50"
  };

  const icons = {
    brain: Brain,
    heart: Heart,
    activity: Activity,
    zap: Zap
  };

  const Icon = icons[icon];

  return (
    <div className="bg-white rounded-lg shadow-sm p-5 border border-gray-100">
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <div className="text-sm text-gray-500 mb-2">{title}</div>
          <div className="flex items-baseline gap-2">
            <span className="text-3xl text-gray-900">{value}</span>
            <span className="text-sm text-gray-500">{unit}</span>
          </div>
        </div>
        <div className={`w-10 h-10 rounded-full flex items-center justify-center ${statusColors[status]}`}>
          <Icon className="w-5 h-5" strokeWidth={1.5} />
        </div>
      </div>
      <div className="mt-3 pt-3 border-t border-gray-100">
        <span className="text-xs text-gray-500 capitalize">{status}</span>
      </div>
    </div>
  );
}
