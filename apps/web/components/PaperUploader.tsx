
import React, { useState } from 'react';
import { Paper, PaperStatus } from '../types';
import { parseRIS, parseBIB } from '../services/parsers';

interface Props {
  onAddPapers: (papers: Paper[]) => void;
  isProcessing: boolean;
  compact?: boolean;
}

const PaperUploader: React.FC<Props> = ({ onAddPapers, isProcessing, compact = false }) => {
  const [dragActive, setDragActive] = useState(false);

  const handleFiles = async (files: FileList) => {
    const allParsedPapers: Paper[] = [];
    
    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      const extension = file.name.split('.').pop()?.toLowerCase();
      
      const content = await file.text();
      
      if (extension === 'ris') {
        allParsedPapers.push(...parseRIS(content, file.name));
      } else if (extension === 'bib' || extension === 'nib') {
        allParsedPapers.push(...parseBIB(content, file.name));
      } else if (extension === 'pdf') {
        const formData = new FormData();
        formData.append('file', file);
        try {
          console.log('[PDF Upload] Sending to backend...');
          const res = await fetch('http://localhost:8000/screening/parse_pdf', {
            method: 'POST',
            body: formData
          });
          const data = await res.json();
          console.log('[PDF Upload] Response:', Object.keys(data));
          
          const fullText = data.text || "";
          const pdfTitle = data.title || file.name.replace('.pdf', '');
          const pdfAbstract = data.abstract || fullText.substring(0, 1500);
          const pdfAuthors = data.authors || "";
          const pdfDoi = data.doi || "";
          const pdfYear = data.year || "";
          const pdfKeywords = data.keywords || [];
          
          if (data.error) {
            console.error('[PDF Upload] Server error:', data.error);
          }
          
          allParsedPapers.push({
            id: Math.random().toString(36).substr(2, 9),
            title: pdfTitle,
            abstract: pdfAbstract.substring(0, 2000) + (pdfAbstract.length > 2000 ? "..." : ""),
            fullText: fullText,
            authors: pdfAuthors,
            doi: pdfDoi,
            year: pdfYear,
            keywords: pdfKeywords,
            status: PaperStatus.PENDING,
            sourceFile: file.name
          });
        } catch(e) {
          console.error('[PDF Upload] Network error:', e);
          allParsedPapers.push({
            id: Math.random().toString(36).substr(2, 9),
            title: file.name.replace('.pdf', ''),
            abstract: "Cannot reach Backend server at localhost:8000. Ensure uvicorn is running.",
            status: PaperStatus.PENDING,
            sourceFile: file.name
          });
        }
      }
    }
    
    onAddPapers(allParsedPapers);
  };

  if (compact) {
    return (
      <div className="bg-white rounded-2xl border border-slate-200 p-4 shadow-sm">
        <h3 className="text-[10px] font-black text-slate-400 uppercase tracking-[0.2em] mb-3">Import More</h3>
        <label className="flex flex-col items-center justify-center border-2 border-dashed border-slate-100 rounded-xl py-6 cursor-pointer hover:bg-slate-50 hover:border-vnu-blue/30 transition-all group">
          <div className="bg-white p-2 rounded-lg shadow-sm mb-2 group-hover:scale-110 transition-transform">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 text-vnu-blue" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M12 4v16m8-8H4" />
            </svg>
          </div>
          <span className="text-[9px] font-black text-slate-400 uppercase tracking-tighter">Add Files</span>
          <input 
            type="file" 
            className="hidden" 
            multiple 
            accept=".pdf,.ris,.bib,.nib" 
            onChange={(e) => e.target.files && handleFiles(e.target.files)}
          />
        </label>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-[32px] shadow-2xl shadow-slate-200/50 border border-slate-200 overflow-hidden">
      <div className="p-10 md:p-16">
        <div 
          className={`relative border-2 border-dashed rounded-[24px] p-12 text-center transition-all group ${
            dragActive ? 'border-vnu-blue bg-vnu-blue/[0.02]' : 'border-slate-200 bg-slate-50/50'
          }`}
          onDragOver={(e) => { e.preventDefault(); setDragActive(true); }}
          onDragLeave={() => setDragActive(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragActive(false);
            if (e.dataTransfer.files) handleFiles(e.dataTransfer.files);
          }}
        >
          <div className="flex flex-col items-center">
            <div className={`p-8 rounded-3xl mb-8 transition-all duration-500 ${dragActive ? 'bg-vnu-blue text-white shadow-2xl shadow-blue-200 rotate-12' : 'bg-white shadow-xl shadow-slate-200/50 text-slate-300 group-hover:text-vnu-blue group-hover:rotate-6'}`}>
              <svg xmlns="http://www.w3.org/2000/svg" className="h-16 w-16" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
              </svg>
            </div>
            <h3 className="text-2xl font-black text-slate-900 mb-3 tracking-tight">Drop your library files</h3>
            <p className="text-slate-500 text-base mb-10 font-medium max-w-sm mx-auto leading-relaxed">
              Drag & drop <span className="text-vnu-blue font-bold">RIS</span>, <span className="text-vnu-blue font-bold">BibTeX</span>, or <span className="text-vnu-blue font-bold">PDF</span> files to start your review.
            </p>
            
            <input 
              type="file" 
              id="file-upload-main" 
              className="hidden" 
              multiple 
              accept=".pdf,.ris,.bib,.nib" 
              onChange={(e) => e.target.files && handleFiles(e.target.files)}
            />
            <label 
              htmlFor="file-upload-main" 
              className="group relative px-12 py-5 bg-vnu-blue text-white rounded-2xl cursor-pointer hover:bg-[#004080] transition-all shadow-xl shadow-blue-200 font-black uppercase tracking-widest text-xs active:scale-95 overflow-hidden"
            >
              <span className="relative z-10 flex items-center gap-3">
                Select Files from Computer
                <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 group-hover:translate-y-0.5 transition-transform" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
                </svg>
              </span>
            </label>
          </div>
        </div>
        
        <div className="mt-12 flex flex-wrap justify-center gap-x-12 gap-y-4 opacity-30 grayscale hover:grayscale-0 hover:opacity-100 transition-all duration-700">
           <div className="flex items-center gap-2">
             <div className="w-2 h-2 rounded-full bg-vnu-blue"></div>
             <span className="text-[10px] font-black uppercase tracking-widest">RIS Format</span>
           </div>
           <div className="flex items-center gap-2">
             <div className="w-2 h-2 rounded-full bg-vnu-yellow"></div>
             <span className="text-[10px] font-black uppercase tracking-widest">BibTeX</span>
           </div>
           <div className="flex items-center gap-2">
             <div className="w-2 h-2 rounded-full bg-vnu-blue"></div>
             <span className="text-[10px] font-black uppercase tracking-widest">PDF Batch</span>
           </div>
        </div>
      </div>
    </div>
  );
};

export default PaperUploader;
