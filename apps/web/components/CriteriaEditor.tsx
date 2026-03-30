
import React from 'react';
import { ReviewCriteria } from '../types';

interface Props {
  criteria: ReviewCriteria;
  onUpdate: (criteria: ReviewCriteria) => void;
}

const CriteriaEditor: React.FC<Props> = ({ criteria, onUpdate }) => {
  const handleChange = (field: keyof ReviewCriteria, value: string) => {
    onUpdate({ ...criteria, [field]: value });
  };

  return (
    <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-5">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-[10px] font-black text-slate-400 uppercase tracking-[0.2em]">
          Review Protocol (PICO)
        </h2>
        <div className="w-6 h-6 rounded-full bg-vnu-blue/10 flex items-center justify-center text-vnu-blue">
          <svg xmlns="http://www.w3.org/2000/svg" className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
          </svg>
        </div>
      </div>
      <div className="space-y-3">
        {[
          { label: 'Population', field: 'population', placeholder: 'e.g., Adults with hypertension' },
          { label: 'Intervention', field: 'intervention', placeholder: 'e.g., Daily meditation' },
          { label: 'Comparison', field: 'comparison', placeholder: 'e.g., No intervention' },
          { label: 'Outcome', field: 'outcome', placeholder: 'e.g., Blood pressure' },
          { label: 'Study Type', field: 'studyType', placeholder: 'e.g., RCT or Cohort' },
        ].map(item => (
          <div key={item.field}>
            <label className="block text-[9px] font-black text-slate-400 uppercase tracking-tighter mb-1 ml-1">{item.label}</label>
            <input 
              className="w-full px-3 py-2 bg-slate-50 border border-slate-200 rounded-xl text-xs focus:ring-2 focus:ring-vnu-blue/20 focus:border-vnu-blue outline-none transition-all placeholder:text-slate-300"
              value={criteria[item.field as keyof ReviewCriteria]}
              onChange={(e) => handleChange(item.field as keyof ReviewCriteria, e.target.value)}
              placeholder={item.placeholder}
            />
          </div>
        ))}
      </div>
    </div>
  );
};

export default CriteriaEditor;
