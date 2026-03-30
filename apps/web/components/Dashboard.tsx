
import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { Paper, PaperStatus } from '../types';

interface Props {
  papers: Paper[];
}

const Dashboard: React.FC<Props> = ({ papers }) => {
  const data = [
    { stage: 'Identified', value: papers.length, color: '#94a3b8' },
    { stage: 'Screened', value: papers.filter(p => p.status !== PaperStatus.PENDING).length, color: '#0054A6' },
    { stage: 'Included (Abs)', value: papers.filter(p => p.status === PaperStatus.ABSTRACT_INCLUDE).length, color: '#10b981' },
    { stage: 'Full-text', value: papers.filter(p => p.status === PaperStatus.FULLTEXT_INCLUDE).length, color: '#059669' },
  ];

  return (
    <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-sm font-black text-slate-800 uppercase tracking-tight">Review Workflow Progress</h3>
        <div className="flex gap-4">
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-emerald-500"></span>
            <span className="text-[10px] font-bold text-slate-500 uppercase">Success Rate: {papers.length ? Math.round((papers.filter(p => p.status === PaperStatus.ABSTRACT_INCLUDE).length / papers.length) * 100) : 0}%</span>
          </div>
        </div>
      </div>
      <div className="h-40">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
            <XAxis dataKey="stage" tick={{ fontSize: 10, fontWeight: 700 }} axisLine={false} tickLine={false} />
            <YAxis hide />
            <Tooltip 
              contentStyle={{ borderRadius: '12px', border: 'none', boxShadow: '0 10px 15px -3px rgb(0 0 0 / 0.1)' }}
              itemStyle={{ fontSize: '12px', fontWeight: 'bold' }}
            />
            <Bar dataKey="value" radius={[6, 6, 0, 0]} barSize={60}>
              {data.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.color} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default Dashboard;
