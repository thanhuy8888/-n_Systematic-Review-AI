
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
      <div className="space-y-4">
        {[
          { 
            label: 'Population', field: 'population', 
            placeholder: 'Vd: Chuột nhắt C57BL/6, DBA/2...', vn: 'Đối tượng (Chuột)',
            suggestions: ['C57BL/6 mice', 'DBA/2 mice', 'Adult male mice', 'Obese mice']
          },
          { 
            label: 'Intervention', field: 'intervention', 
            placeholder: 'Vd: HFD, HFHS, Western Diet...', vn: 'Chế độ ăn can thiệp',
            suggestions: ['High-Fat Diet (HFD)', 'High-Sugar Diet', 'HFHS (Western Diet)', 'High Cholesterol Diet']
          },
          { 
            label: 'Comparison', field: 'comparison', 
            placeholder: 'Vd: Standard chow, Low-fat diet...', vn: 'Chế độ ăn đối chứng',
            suggestions: ['Standard chow diet', 'Purified low-fat diet', 'Pair-fed control']
          },
          { 
            label: 'Outcome', field: 'outcome', 
            placeholder: 'Vd: Body weight, HOMA-IR, Steatosis...', vn: 'Chỉ số chuyển hóa',
            suggestions: ['Body weight & Adiposity', 'HOMA-IR', 'Lipotoxic Steatosis', 'TC/TG/LDL-C', 'ALT/AST Liver enzymes']
          },
          { 
            label: 'Study Type', field: 'studyType', 
            placeholder: 'Vd: In vivo, Randomized...', vn: 'Loại hình nghiên cứu',
            suggestions: ['In vivo mouse model', 'Randomized controlled trial']
          },
        ].map(item => (
          <div key={item.field} className="relative">
            <div className="flex items-center gap-1.5 mb-1.5 ml-1">
              <label className="block text-[10px] font-black text-slate-500 uppercase tracking-tighter">{item.label}</label>
              <span className="text-[9px] text-slate-400 font-medium normal-case tracking-normal">({item.vn})</span>
            </div>
            
            <input 
              className="w-full px-3 py-2.5 bg-slate-50 border border-slate-200 rounded-xl text-xs focus:ring-2 focus:ring-vnu-blue/20 focus:border-vnu-blue outline-none transition-all placeholder:text-slate-300 font-medium shadow-inner"
              value={criteria[item.field as keyof ReviewCriteria]}
              onChange={(e) => handleChange(item.field as keyof ReviewCriteria, e.target.value)}
              placeholder={item.placeholder}
            />

            {/* AI Smart Chips */}
            <div className="flex flex-wrap gap-1.5 mt-2 ml-1">
               <div className="flex items-center gap-1 text-[8px] font-black uppercase tracking-widest text-blue-500 mr-1 mt-0.5">
                  ✨ Gợi ý:
               </div>
               {item.suggestions.map(sug => (
                 <button 
                   key={sug}
                   onClick={() => {
                     const currentVal = criteria[item.field as keyof ReviewCriteria] as string;
                     const newVal = currentVal ? `${currentVal}, ${sug}` : sug;
                     handleChange(item.field as keyof ReviewCriteria, newVal);
                   }}
                   className="text-[9px] font-semibold text-slate-600 bg-white border border-slate-200 hover:border-blue-400 hover:text-blue-600 hover:bg-blue-50 px-2 py-0.5 rounded-md transition-colors active:scale-95"
                 >
                   {sug}
                 </button>
               ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default CriteriaEditor;
