import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  CartesianGrid,
  Legend,
} from "recharts";
import { useI18n } from '../i18n/I18nContext';

interface RiskTimelineChartProps {
  probabilities: number[];
  threshold: number;
}

const RiskTimelineChart: React.FC<RiskTimelineChartProps> = ({ probabilities, threshold }) => {
  const { t } = useI18n();

  const data = probabilities.map((p, idx) => ({
    frame: idx,
    prob: p,
  }));

  return (
    <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-2">
      <h2 className="text-lg font-semibold">{t("riskTimelineTitle")}</h2>
      <p className="text-xs text-slate-400">{t("riskTimelineSubtitle")}</p>
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 10, right: 20, left: 0, bottom: 20 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1f2933" />
            <XAxis
              dataKey="frame"
              tick={{ fontSize: 10, fill: "#9ca3af" }}
              label={{ value: t("xAxisFramesLabel"), position: "insideBottom", offset: -5, fill: "#9ca3af" }}
            />
            <YAxis
              domain={[0, 1]}
              tick={{ fontSize: 10, fill: "#9ca3af" }}
              label={{ value: t("yAxisProbabilityLabel"), angle: -90, position: "insideLeft", fill: "#9ca3af" }}
            />
            <Tooltip
              contentStyle={{ backgroundColor: "#020617", borderColor: "#1f2937" }}
              labelStyle={{ color: "#e5e7eb" }}
            />
            <Legend wrapperStyle={{ fontSize: 10, color: "#9ca3af" }} />
            <ReferenceLine
              y={threshold}
              stroke="#f97373"
              strokeDasharray="4 4"
              label={{
                value: t("thresholdLegendLabel"),
                position: "insideTopRight",
                fill: "#fca5a5",
                fontSize: 10,
              }}
            />
            <Line type="monotone" dataKey="prob" stroke="#38bdf8" strokeWidth={2} dot={false} name="P(collision)" />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default RiskTimelineChart;
