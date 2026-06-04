
import React from 'react';
import { Paper, PaperStatus } from '../types';

interface Props {
  papers: Paper[];
}

const Dashboard: React.FC<Props> = ({ papers }) => {
  const identified = papers.length;
  const duplicatesRemoved = 0; // tracked in future dedup step
  const screened = papers.filter(p => p.status !== PaperStatus.PENDING).length;
  const abstractExcluded = papers.filter(p => p.status === PaperStatus.ABSTRACT_EXCLUDE).length;
  const abstractIncluded = papers.filter(p =>
    p.status === PaperStatus.ABSTRACT_INCLUDE ||
    p.status === PaperStatus.FULLTEXT_INCLUDE ||
    p.status === PaperStatus.FULLTEXT_EXCLUDE ||
    p.status === PaperStatus.EXTRACTED
  ).length;
  const fulltextAssessed = papers.filter(p =>
    p.status === PaperStatus.FULLTEXT_INCLUDE ||
    p.status === PaperStatus.FULLTEXT_EXCLUDE ||
    p.status === PaperStatus.EXTRACTED
  ).length;
  const fulltextExcluded = papers.filter(p => p.status === PaperStatus.FULLTEXT_EXCLUDE).length;
  const included = papers.filter(p =>
    p.status === PaperStatus.FULLTEXT_INCLUDE || p.status === PaperStatus.EXTRACTED
  ).length;
  const extracted = papers.filter(p => p.status === PaperStatus.EXTRACTED).length;

  const pendingCount = papers.filter(p => p.status === PaperStatus.PENDING).length;
  const maybeCount = papers.filter(p => p.status === PaperStatus.MAYBE).length;

  return (
    <div className="bg-white p-10 rounded-[32px] shadow-lg border border-slate-200 space-y-8 max-w-4xl mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-slate-100 pb-4">
        <h3 className="text-base font-black text-slate-800 uppercase tracking-tight">PRISMA Flow — Review Progress</h3>
        <span className="text-xs font-bold text-slate-400 uppercase tracking-wider">
          {extracted}/{included} studies extracted
        </span>
      </div>

      {/* PRISMA Flow Diagram */}
      <div className="flex flex-col items-center gap-0 select-none py-4">

        {/* Row 1: Identification */}
        <PrismaBox
          label="Records Identified"
          count={identified}
          color="bg-slate-100 border-slate-300 text-slate-700"
          dotColor="bg-slate-400"
        />
        <Arrow />

        {/* Row 2: After dedup */}
        <div className="flex items-center gap-6 w-full max-w-2xl justify-center">
          <PrismaBox
            label="Records After Deduplication"
            count={identified - duplicatesRemoved}
            color="bg-blue-50 border-blue-200 text-blue-800"
            dotColor="bg-blue-400"
            small
          />
          <ExclusionBox label="Duplicates Removed" count={duplicatesRemoved} />
        </div>
        <Arrow />

        {/* Row 3: Screened */}
        <div className="flex items-center gap-6 w-full max-w-2xl justify-center">
          <PrismaBox
            label="Titles/Abstracts Screened"
            count={screened}
            color="bg-indigo-50 border-indigo-200 text-indigo-800"
            dotColor="bg-indigo-400"
            small
          />
          <ExclusionBox label="Excluded at Abstract" count={abstractExcluded} />
        </div>
        <Arrow />

        {/* Row 4: Full-text */}
        <div className="flex items-center gap-6 w-full max-w-2xl justify-center">
          <PrismaBox
            label="Full-text Assessed for Eligibility"
            count={fulltextAssessed}
            color="bg-amber-50 border-amber-200 text-amber-800"
            dotColor="bg-amber-400"
            small
          />
          <ExclusionBox label="Excluded at Full-text" count={fulltextExcluded} />
        </div>
        <Arrow />

        {/* Row 5: Included */}
        <PrismaBox
          label="Studies Included in Synthesis"
          count={included}
          color="bg-emerald-50 border-emerald-400 text-emerald-900"
          dotColor="bg-emerald-500"
          highlight
        />
      </div>

      {/* Quick Stats Row */}
      <div className="grid grid-cols-4 gap-4 pt-4 border-t border-slate-100">
        <StatChip label="Pending" count={pendingCount} color="text-slate-500 bg-slate-100" />
        <StatChip label="Maybe" count={maybeCount} color="text-amber-600 bg-amber-50" />
        <StatChip label="Extracted" count={extracted} color="text-emerald-700 bg-emerald-50" />
        <StatChip
          label="Include Rate"
          count={`${identified ? Math.round((abstractIncluded / identified) * 100) : 0}%`}
          color="text-blue-700 bg-blue-50"
        />
      </div>
    </div>
  );
};

const Arrow: React.FC = () => (
  <div className="flex flex-col items-center py-1">
    <div className="w-px h-8 bg-slate-300"></div>
    <svg className="w-4 h-4 text-slate-400 -mt-1" viewBox="0 0 12 12" fill="currentColor">
      <path d="M6 10L1 4h10z" />
    </svg>
  </div>
);

const PrismaBox: React.FC<{
  label: string;
  count: number;
  color: string;
  dotColor: string;
  small?: boolean;
  highlight?: boolean;
}> = ({ label, count, color, dotColor, small, highlight }) => (
  <div className={`flex items-center justify-between border rounded-2xl px-6 py-4 w-full max-w-sm ${color} ${highlight ? 'shadow-md ring-1 ring-emerald-500/20' : ''}`}>
    <div className="flex items-center gap-3">
      <div className={`w-2.5 h-2.5 rounded-full ${dotColor}`}></div>
      <span className={`font-bold ${small ? 'text-xs' : 'text-sm'}`}>{label}</span>
    </div>
    <span className={`font-black ${highlight ? 'text-xl' : 'text-lg'} ml-4`}>{count}</span>
  </div>
);

const ExclusionBox: React.FC<{ label: string; count: number }> = ({ label, count }) => (
  <div className="flex items-center gap-3 border border-rose-200 bg-rose-50 rounded-2xl px-5 py-3 min-w-[180px]">
    <div className="w-2.5 h-2.5 rounded-full bg-rose-400 shrink-0"></div>
    <div className="flex-1">
      <div className="text-xs font-bold text-rose-700 leading-tight">{label}</div>
      <div className="text-base font-black text-rose-600">{count}</div>
    </div>
  </div>
);

const StatChip: React.FC<{ label: string; count: number | string; color: string }> = ({
  label, count, color
}) => (
  <div className={`rounded-2xl px-5 py-3 text-center ${color}`}>
    <div className="text-xl font-black">{count}</div>
    <div className="text-xs font-bold uppercase tracking-wider opacity-70">{label}</div>
  </div>
);

export default Dashboard;
