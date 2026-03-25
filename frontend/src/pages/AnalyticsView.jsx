import React, { useEffect, useState } from "react";
import client from "../api/client";
import { motion } from "framer-motion";
import { Loader2, Activity, HardDrive, Target } from "lucide-react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  Legend,
  Cell
} from "recharts";

function StatCard({ label, value, icon: Icon, delay = 0 }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 15 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay, duration: 0.4 }}
      className="glass-card p-6 flex items-start justify-between group"
    >
      <div>
        <p className="text-zinc-400 font-medium text-sm mb-1 uppercase tracking-wider">{label}</p>
        <h3 className="text-3xl font-semibold text-white tracking-tight">{value}</h3>
      </div>
      <div className="p-3 bg-white/5 rounded-xl border border-white/5 transition-transform group-hover:scale-110">
        <Icon className="w-5 h-5 text-zinc-300" />
      </div>
    </motion.div>
  );
}

const CustomTooltip = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    return (
      <div className="glass-card p-3 border border-white/10 shadow-2xl">
        <p className="font-semibold text-sm text-white mb-1">{label}</p>
        {payload.map((entry, idx) => (
          <div key={idx} className="flex flex-col">
            <span className="text-xs text-zinc-400 uppercase tracking-widest">{entry.name}</span>
            <span className="text-[13px] font-medium text-emerald-400">{Number(entry.value).toFixed(2)}</span>
          </div>
        ))}
      </div>
    );
  }
  return null;
};

export default function AnalyticsView() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    async function fetchDash() {
      try {
        const res = await client.get("/eval/dashboard");
        setData(res.data);
      } catch (err) {
        setError("Failed to fetch analytics data.");
      } finally {
        setLoading(false);
      }
    }
    fetchDash();
  }, []);

  if (loading) {
    return (
      <div className="h-full flex items-center justify-center">
        <Loader2 className="w-8 h-8 animate-spin text-zinc-500" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-12 text-center text-red-400">
        <p>{error}</p>
      </div>
    );
  }

  // Parse RAGAS data
  const ragas = data.ragas || {};
  
  // Format data for radar chart to visualize RAG performance dimensions
  const radarData = [
    { subject: 'Context Precision', A: ragas.context_precision || 0, fullMark: 1 },
    { subject: 'Recall', A: ragas.context_recall || 0, fullMark: 1 },
    { subject: 'Faithfulness', A: ragas.faithfulness || 0, fullMark: 1 },
    { subject: 'Answer Relevancy', A: ragas.answer_relevancy || 0, fullMark: 1 },
  ];

  // Format dataset for a simple bar chart of the individual queries if they existed (not provided directly, but let's mock the overall metrics as a bar chart comparison if needed)
  const barData = radarData.map(d => ({ name: d.subject, Score: d.A }));

  return (
    <div className="max-w-6xl mx-auto p-8 lg:p-12 w-full">
      <div className="mb-10">
        <h1 className="text-3xl font-semibold tracking-tight text-white mb-2">Analytics</h1>
        <p className="text-zinc-400">View performance metrics and RAG evaluation scores.</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
        <StatCard
          icon={Activity}
          label="Total Queries"
          value={data.total_queries}
          delay={0}
        />
        <StatCard
          icon={HardDrive}
          label="Cache Hit Rate"
          value={`${(data.cache_hit_rate * 100).toFixed(1)}%`}
          delay={0.1}
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Radar Chart */}
        <motion.div
          initial={{ opacity: 0, y: 15 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2, duration: 0.4 }}
          className="glass-card p-6 h-[400px] flex flex-col"
        >
          <div className="flex items-center gap-2 mb-6">
            <Target className="w-4 h-4 text-emerald-400" />
            <h3 className="font-medium text-white">RAGAS Quality Matrix</h3>
          </div>
          <div className="flex-1 w-full min-h-0">
            <ResponsiveContainer width="100%" height="100%">
              <RadarChart outerRadius="70%" data={radarData}>
                <PolarGrid stroke="rgba(255,255,255,0.1)" />
                <PolarAngleAxis dataKey="subject" tick={{ fill: '#A1A1AA', fontSize: 11, fontWeight: 500 }} />
                <PolarRadiusAxis angle={30} domain={[0, 1]} tick={{ fill: 'transparent' }} axisLine={false} />
                <Tooltip content={<CustomTooltip />} />
                <Radar name="Hermes Engine" dataKey="A" stroke="#10b981" fill="#10b981" fillOpacity={0.15} />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </motion.div>

        {/* Bar Chart */}
        <motion.div
          initial={{ opacity: 0, y: 15 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3, duration: 0.4 }}
          className="glass-card p-6 h-[400px] flex flex-col"
        >
          <div className="flex items-center gap-2 mb-6">
            <BarChart className="w-4 h-4 text-zinc-300" />
            <h3 className="font-medium text-white">Evaluation Score Breakdown</h3>
          </div>
          <div className="flex-1 w-full min-h-0">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={barData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                <XAxis dataKey="name" stroke="#52525b" tick={{ fill: '#A1A1AA', fontSize: 11 }} axisLine={false} tickLine={false} dy={10} />
                <YAxis stroke="#52525b" domain={[0, 1]} tick={{ fill: '#52525b', fontSize: 11 }} axisLine={false} tickLine={false} />
                <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(255,255,255,0.02)' }} />
                <Bar dataKey="Score" radius={[4, 4, 0, 0]}>
                  {barData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.Score > 0.8 ? '#10b981' : entry.Score > 0.5 ? '#eab308' : '#ef4444'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </motion.div>
      </div>

    </div>
  );
}
