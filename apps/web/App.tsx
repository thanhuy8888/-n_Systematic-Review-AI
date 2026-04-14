
import React, { useState, useMemo, useEffect } from 'react';
import { Paper, PaperStatus, ReviewCriteria, ReviewStage } from './types';
import { screenPaper, extractPaperData } from './services/mlService';
import { saveProject, loadProject, exportProjectFile } from './services/storageService';
import PaperUploader from './components/PaperUploader';
import CriteriaEditor from './components/CriteriaEditor';
import Dashboard from './components/Dashboard';

const ITEMS_PER_PAGE = 20;

const App: React.FC = () => {
  const [papers, setPapers] = useState<Paper[]>([]);
  const [activeStage, setActiveStage] = useState<ReviewStage>(ReviewStage.ABSTRACT_SCREENING);
  const [isProcessing, setIsProcessing] = useState(false);
  const [currentPage, setCurrentPage] = useState(1);
  const [searchQuery, setSearchQuery] = useState('');
  const [hasLoaded, setHasLoaded] = useState(false);
  const [selectedPaperId, setSelectedPaperId] = useState<string | null>(null);
  const [criteria, setCriteria] = useState<ReviewCriteria>({
    population: 'Adults with hypertension',
    intervention: 'Daily meditation practice',
    comparison: 'No intervention',
    outcome: 'Reduced blood pressure',
    studyType: 'RCT or cohort studies'
  });

  // Sidebar Filter States
  const [decisionFilter, setDecisionFilter] = useState<string>('ALL');

  useEffect(() => {
    loadProject().then(data => {
      if (data) {
        setPapers(data.papers);
        setCriteria(data.criteria);
        // Nếu đã có bài báo, tự động chuyển sang Abstract Screening
        if (data.papers.length > 0) setActiveStage(ReviewStage.ABSTRACT_SCREENING);
        else setActiveStage(ReviewStage.IDENTIFICATION);
      } else {
        setActiveStage(ReviewStage.IDENTIFICATION);
      }
      setHasLoaded(true);
    });
  }, []);

  useEffect(() => {
    if (hasLoaded && papers.length > 0) {
      saveProject(papers, criteria);
    }
  }, [papers, criteria, hasLoaded]);

  const addPapers = (newPapers: Paper[]) => {
    setPapers(prev => [...prev, ...newPapers]);
    setActiveStage(ReviewStage.ABSTRACT_SCREENING);
  };

  const filteredPapers = useMemo(() => {
    let result = papers;

    if (activeStage === ReviewStage.ABSTRACT_SCREENING) {
      // Abstract stage focus
    } else if (activeStage === ReviewStage.FULLTEXT_SCREENING) {
      result = result.filter(p => p.status === PaperStatus.ABSTRACT_INCLUDE || p.status === PaperStatus.FULLTEXT_INCLUDE || p.status === PaperStatus.FULLTEXT_EXCLUDE);
    } else if (activeStage === ReviewStage.EXTRACTION) {
      result = result.filter(p => p.status === PaperStatus.FULLTEXT_INCLUDE || p.status === PaperStatus.EXTRACTED);
    }

    if (decisionFilter !== 'ALL') {
      result = result.filter(p => p.status === decisionFilter);
    }

    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase();
      result = result.filter(p => 
        p.title.toLowerCase().includes(q) || 
        p.abstract.toLowerCase().includes(q) ||
        p.authors?.toLowerCase().includes(q)
      );
    }

    return result;
  }, [papers, activeStage, decisionFilter, searchQuery]);

  const paginatedPapers = useMemo(() => {
    const start = (currentPage - 1) * ITEMS_PER_PAGE;
    return filteredPapers.slice(start, start + ITEMS_PER_PAGE);
  }, [filteredPapers, currentPage]);

  // Handle global keyboard shortcuts and auto-selection
  useEffect(() => {
    if (paginatedPapers.length > 0 && !selectedPaperId) {
      setSelectedPaperId(paginatedPapers[0].id);
    } else if (paginatedPapers.length === 0) {
      setSelectedPaperId(null);
    }
  }, [paginatedPapers, selectedPaperId]);

  useEffect(() => {
    if (activeStage === ReviewStage.IDENTIFICATION) return;
    
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ignore if user is typing in inputs (like search)
      if (document.activeElement?.tagName === 'INPUT' || document.activeElement?.tagName === 'TEXTAREA') return;
      
      const currentIndex = filteredPapers.findIndex(p => p.id === selectedPaperId);
      if (currentIndex === -1) return;

      const goNext = () => {
        if (currentIndex < filteredPapers.length - 1) {
          const nextId = filteredPapers[currentIndex + 1].id;
          setSelectedPaperId(nextId);
          // Auto change page block if moving past current paginated view
          const nextPageIndex = Math.floor((currentIndex + 1) / ITEMS_PER_PAGE) + 1;
          if (nextPageIndex !== currentPage) setCurrentPage(nextPageIndex);
        }
      };

      const goPrev = () => {
        if (currentIndex > 0) {
          const prevId = filteredPapers[currentIndex - 1].id;
          setSelectedPaperId(prevId);
          const prevPageIndex = Math.floor((currentIndex - 1) / ITEMS_PER_PAGE) + 1;
          if (prevPageIndex !== currentPage) setCurrentPage(prevPageIndex);
        }
      };

      if (e.key === 'ArrowDown') {
        e.preventDefault();
        goNext();
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        goPrev();
      } else if (e.key.toLowerCase() === 'i') {
        e.preventDefault();
        updatePaperStatus(selectedPaperId!, PaperStatus.ABSTRACT_INCLUDE);
        goNext();
      } else if (e.key.toLowerCase() === 'e') {
        e.preventDefault();
        updatePaperStatus(selectedPaperId!, PaperStatus.ABSTRACT_EXCLUDE);
        goNext();
      } else if (e.key.toLowerCase() === 'm') {
        e.preventDefault();
        updatePaperStatus(selectedPaperId!, PaperStatus.MAYBE);
        goNext();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [activeStage, selectedPaperId, filteredPapers, currentPage]);

  const updatePaperStatus = (id: string, status: PaperStatus) => {
    setPapers(prev => prev.map(p => p.id === id ? { ...p, status } : p));
  };

  const handleRemovePaper = (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    // Papers in this local UI version are stored in IndexedDB browser storage (via saveProject effect)
    // rather than relying on the backend, so we just filter it from the React state directly.
    setPapers(prev => prev.filter(p => p.id !== id));
  };

  const handleBatchAIScreening = async () => {
    let targetPapers: typeof filteredPapers = [];
    let stageLabel = "";
    
    if (activeStage === ReviewStage.ABSTRACT_SCREENING) {
      targetPapers = filteredPapers.filter(p => p.status === PaperStatus.PENDING);
      stageLabel = "Abstract Screening";
    } else if (activeStage === ReviewStage.FULLTEXT_SCREENING) {
      targetPapers = filteredPapers.filter(p => p.status === PaperStatus.ABSTRACT_INCLUDE);
      stageLabel = "Full-text Screening";
    } else if (activeStage === ReviewStage.EXTRACTION) {
      targetPapers = filteredPapers.filter(p => !p.extractionData && (p.status === PaperStatus.ABSTRACT_INCLUDE || p.status === PaperStatus.FULLTEXT_INCLUDE || p.status === PaperStatus.EXTRACTED));
      stageLabel = "Extraction";
    } else {
      alert("Vui lòng chuyển sang tab Abstract Screening, Full-text Screening, hoặc Extraction để chạy AI Agent.");
      return;
    }

    if (targetPapers.length === 0) {
      alert(`Không có bài báo nào cần AI xử lý ở vòng ${stageLabel} (đã xử lý xong hoặc rỗng).`);
      return;
    }

    setIsProcessing(true);
    const chunkSize = 5;
    const updatedPapers = [...papers];

    for (let i = 0; i < targetPapers.length; i += chunkSize) {
      const chunk = targetPapers.slice(i, i + chunkSize);
      await Promise.all(chunk.map(async (paper) => {
        if (activeStage === ReviewStage.EXTRACTION) {
           const extractedPayload = await extractPaperData(paper);
           const index = updatedPapers.findIndex(p => p.id === paper.id);
           if (index !== -1 && extractedPayload) {
             updatedPapers[index] = { 
               ...updatedPapers[index], 
               extractionData: extractedPayload,
               status: PaperStatus.EXTRACTED
             };
           }
        } else if (activeStage === ReviewStage.FULLTEXT_SCREENING) {
           // Full-text screening: send the paper with fullText — backend auto-detects
           console.log(`[Full-text] Screening: ${paper.title.substring(0, 50)}, fullText length: ${paper.fullText?.length || 0}`);
           const result = await screenPaper(paper, criteria);
           const index = updatedPapers.findIndex(p => p.id === paper.id);
           if (index !== -1) {
             const ftInclude = result.status === PaperStatus.ABSTRACT_INCLUDE;
             const newStatus = ftInclude ? PaperStatus.FULLTEXT_INCLUDE : PaperStatus.FULLTEXT_EXCLUDE;
             // Preserve previous AI screening reason + append full-text result
             const prevReason = updatedPapers[index].aiScreeningReason || '';
             updatedPapers[index] = { 
               ...updatedPapers[index], 
               status: newStatus, 
               aiScreeningReason: prevReason + `\n--- [Full-text Screening] ---\n${result.reason}` 
             };
           }
        } else {
           // Abstract screening: send ONLY abstract (strip fullText for quick filtering)
           const abstractOnly = { ...paper, fullText: undefined };
           const result = await screenPaper(abstractOnly, criteria);
           const index = updatedPapers.findIndex(p => p.id === paper.id);
           if (index !== -1) {
             updatedPapers[index] = { 
               ...updatedPapers[index], 
               status: result.status, 
               aiScreeningReason: result.reason 
             };
           }
        }
      }));
      setPapers([...updatedPapers]);
      await new Promise(r => setTimeout(r, 400));
    }
    setIsProcessing(false);
  };

  const handleImportProject = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (event) => {
      try {
        const data = JSON.parse(event.target?.result as string);
        if (data.papers && data.criteria) {
          setPapers(data.papers);
          setCriteria(data.criteria);
          setActiveStage(ReviewStage.ABSTRACT_SCREENING);
        }
      } catch (err) {
        alert("Invalid project file.");
      }
    };
    reader.readAsText(file);
  };

  return (
    <div className="min-h-screen bg-white flex flex-col font-['Inter']">
      <header className="bg-vnu-blue text-white shadow-md z-50">
        <div className="max-w-[1600px] mx-auto px-4 h-14 flex items-center justify-between">
          <div className="flex items-center gap-6">
            <div className="flex items-center gap-2">
              <div className="bg-vnu-yellow p-1.5 rounded-md shadow-sm">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-vnu-blue" viewBox="0 0 20 20" fill="currentColor">
                  <path d="M9 4.804A7.994 7.994 0 002 12a7.994 7.994 0 007 7.196V4.804zM11 4.804v14.392A7.994 7.994 0 0018 12a7.994 7.994 0 00-7-7.196z" />
                </svg>
              </div>
              <span className="font-bold text-xl tracking-tight">IS <span className="text-vnu-yellow font-light ml-1">Systematic Review</span></span>
            </div>

            <nav className="flex h-14 ml-8">
              {Object.values(ReviewStage).map((stage, idx) => (
                <button
                  key={stage}
                  onClick={() => { setActiveStage(stage); setCurrentPage(1); }}
                  className={`px-5 h-full text-[11px] uppercase tracking-wider font-bold transition-all relative ${
                    activeStage === stage ? 'text-vnu-yellow' : 'text-blue-100 hover:text-white'
                  }`}
                >
                  <span className="relative z-10">{stage}</span>
                  {activeStage === stage && (
                    <div className="absolute bottom-0 left-0 w-full h-1 bg-vnu-yellow rounded-t-full shadow-[0_-2px_10px_rgba(253,184,19,0.5)]"></div>
                  )}
                </button>
              ))}
            </nav>
          </div>

          <div className="flex items-center gap-4">
            <div className="flex bg-white/10 p-1 rounded-lg">
              <button onClick={() => exportProjectFile(papers, criteria)} className="px-3 py-1 text-xs font-bold hover:bg-white/10 rounded">Export</button>
              <label className="px-3 py-1 text-xs font-bold hover:bg-white/10 rounded cursor-pointer">
                Import <input type="file" className="hidden" accept=".json" onChange={handleImportProject} />
              </label>
            </div>
            <button 
              onClick={handleBatchAIScreening}
              disabled={isProcessing || activeStage === ReviewStage.IDENTIFICATION}
              className="bg-vnu-yellow hover:bg-[#e5a611] text-vnu-blue px-4 py-1.5 rounded text-xs font-bold uppercase tracking-widest shadow-lg transition-all active:scale-95 disabled:opacity-50"
            >
              {isProcessing ? 'AI Working...' : 
                activeStage === ReviewStage.ABSTRACT_SCREENING ? '🔬 Abstract Screening' :
                activeStage === ReviewStage.FULLTEXT_SCREENING ? '📄 Full-text Screening' :
                activeStage === ReviewStage.EXTRACTION ? '📊 Run Extraction' :
                'Run AI Agent'}
            </button>
          </div>
        </div>
      </header>

      <div className="flex flex-1 overflow-hidden">
        {/* Sidebar */}
        <aside className="w-72 bg-slate-50 border-r border-slate-200 overflow-y-auto p-5 space-y-8">
          <div>
            <h3 className="text-[10px] font-black text-slate-400 uppercase tracking-[0.2em] mb-4">Review Decisions</h3>
            <div className="space-y-1.5">
              {[
                { label: 'All Studies', value: 'ALL', count: papers.length, icon: 'M4 6h16M4 10h16M4 14h16M4 18h16' },
                { label: 'Included', value: PaperStatus.ABSTRACT_INCLUDE, count: papers.filter(p => p.status === PaperStatus.ABSTRACT_INCLUDE).length, color: 'text-emerald-600', bg: 'bg-emerald-50', icon: 'M5 13l4 4L19 7' },
                { label: 'Excluded', value: PaperStatus.ABSTRACT_EXCLUDE, count: papers.filter(p => p.status === PaperStatus.ABSTRACT_EXCLUDE).length, color: 'text-rose-600', bg: 'bg-rose-50', icon: 'M6 18L18 6M6 6l12 12' },
                { label: 'Maybe', value: PaperStatus.MAYBE, count: papers.filter(p => p.status === PaperStatus.MAYBE).length, color: 'text-amber-600', bg: 'bg-amber-50', icon: 'M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z' },
                { label: 'Undecided', value: PaperStatus.PENDING, count: papers.filter(p => p.status === PaperStatus.PENDING).length, color: 'text-slate-500', bg: 'bg-slate-100', icon: 'M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z' },
              ].map(opt => (
                <button 
                  key={opt.value}
                  onClick={() => setDecisionFilter(opt.value)}
                  className={`flex items-center justify-between w-full px-3 py-2.5 rounded-xl text-xs font-semibold transition-all group ${
                    decisionFilter === opt.value ? 'bg-white shadow-sm border border-slate-200 text-vnu-blue' : 'text-slate-600 hover:bg-slate-200/50'
                  }`}
                >
                  <div className="flex items-center gap-3">
                    <div className={`p-1.5 rounded-lg ${decisionFilter === opt.value ? 'bg-vnu-blue text-white' : opt.bg + ' ' + opt.color}`}>
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d={opt.icon} />
                      </svg>
                    </div>
                    <span>{opt.label}</span>
                  </div>
                  <span className={`px-2 py-0.5 rounded-full text-[10px] font-bold ${decisionFilter === opt.value ? 'bg-vnu-blue text-white' : 'bg-slate-200 text-slate-500'}`}>
                    {opt.count}
                  </span>
                </button>
              ))}
            </div>
          </div>

          <CriteriaEditor criteria={criteria} onUpdate={setCriteria} />
          
          {/* Uploader in sidebar only as a fallback or if papers already exist */}
          {papers.length > 0 && <PaperUploader onAddPapers={addPapers} isProcessing={isProcessing} compact />}

          <div>
            <h3 className="text-[10px] font-black text-slate-400 uppercase tracking-widest mb-3">Keywords for Inclusion</h3>
            <div className="flex flex-wrap gap-1">
              {['RCT', 'Human', 'Clinical trial', 'Hypertension'].map(kw => (
                <span key={kw} onClick={() => setCriteria(prev => ({...prev, intervention: prev.intervention + (prev.intervention ? ', ' : '') + kw}))} className="bg-emerald-50 text-emerald-700 text-[10px] px-2 py-0.5 rounded border border-emerald-100 cursor-pointer hover:bg-emerald-100">
                  {kw}
                </span>
              ))}
            </div>
          </div>
        </aside>

        {/* Main Workspace */}
        <main className="flex-1 overflow-y-auto p-6">
          <div className="max-w-5xl mx-auto space-y-6">
            {activeStage === ReviewStage.IDENTIFICATION ? (
              <div className="flex flex-col items-center justify-center min-h-[70vh] py-12">
                <div className="w-full max-w-3xl">
                   <div className="text-center mb-12">
                     <div className="inline-block p-4 bg-vnu-blue/5 rounded-3xl mb-6">
                        <div className="bg-vnu-blue p-4 rounded-2xl shadow-xl shadow-blue-200">
                          <svg xmlns="http://www.w3.org/2000/svg" className="h-10 w-10 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                          </svg>
                        </div>
                     </div>
                     <h2 className="text-4xl font-black text-slate-900 mb-4 tracking-tight">Intelligent Evidence Discovery</h2>
                     <p className="text-slate-500 text-lg max-w-xl mx-auto leading-relaxed">
                       Accelerate your systematic review with AI-powered screening and extraction. 
                       Upload your study library to begin.
                     </p>
                   </div>
                   
                   <div className="relative">
                      <div className="absolute -inset-4 bg-gradient-to-r from-vnu-blue/5 to-vnu-yellow/5 rounded-[40px] blur-2xl -z-10"></div>
                      <PaperUploader onAddPapers={addPapers} isProcessing={isProcessing} />
                   </div>
                   
                   {papers.length > 0 && (
                     <div className="mt-12 p-8 bg-white rounded-[32px] border border-slate-200 shadow-2xl shadow-slate-200/50 text-center relative overflow-hidden group">
                       <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-vnu-blue via-vnu-yellow to-vnu-blue"></div>
                       <div className="flex items-center justify-center gap-3 mb-6">
                          <div className="flex -space-x-2">
                            {[1,2,3].map(i => (
                              <div key={i} className="w-8 h-8 rounded-full bg-slate-100 border-2 border-white flex items-center justify-center text-[10px] font-bold text-slate-400">
                                <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" /></svg>
                              </div>
                            ))}
                          </div>
                          <p className="text-sm font-bold text-slate-700">
                            <span className="text-vnu-blue text-lg">{papers.length}</span> studies ready for screening
                          </p>
                       </div>
                       <button 
                        onClick={() => setActiveStage(ReviewStage.ABSTRACT_SCREENING)}
                        className="group relative bg-vnu-blue text-white px-12 py-4 rounded-2xl font-black uppercase tracking-widest hover:bg-[#004080] transition-all shadow-xl shadow-blue-200 active:scale-95 overflow-hidden"
                       >
                         <span className="relative z-10 flex items-center gap-3">
                           Start Screening Process
                           <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 group-hover:translate-x-1 transition-transform" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                             <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                           </svg>
                         </span>
                       </button>
                     </div>
                   )}
                </div>
              </div>
            ) : (
              <>
                <Dashboard papers={papers} />

                <div className="flex items-center justify-between bg-white px-6 py-3 rounded-xl border border-slate-200 shadow-sm">
                  <div className="relative flex-1 max-w-md">
                    <input 
                      type="text" 
                      placeholder="Search in results..." 
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      className="w-full pl-9 pr-4 py-1.5 bg-slate-50 border border-slate-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-vnu-blue"
                    />
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 absolute left-3 top-2.5 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                    </svg>
                  </div>
                  <div className="flex items-center gap-2 text-xs font-bold text-slate-500">
                    <span>Show {filteredPapers.length} results</span>
                    <div className="flex items-center bg-slate-100 rounded p-0.5 ml-4">
                      <button disabled={currentPage === 1} onClick={() => setCurrentPage(p => p - 1)} className="p-1 hover:bg-white rounded disabled:opacity-30">
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" /></svg>
                      </button>
                      <span className="px-3">Page {currentPage}</span>
                      <button disabled={currentPage * ITEMS_PER_PAGE >= filteredPapers.length} onClick={() => setCurrentPage(p => p + 1)} className="p-1 hover:bg-white rounded disabled:opacity-30">
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" /></svg>
                      </button>
                    </div>
                  </div>
                </div>

                <div className="flex gap-6 h-[72vh] min-h-[600px] mt-6">
                  {/* Left Column: Dense List View */}
                  <div className="w-[38%] bg-white border border-slate-200 rounded-2xl overflow-y-auto shadow-sm flex flex-col hide-scrollbar relative">
                     {paginatedPapers.map(paper => (
                        <div 
                           key={paper.id} 
                           onClick={() => setSelectedPaperId(paper.id)}
                           className={`p-4 border-b border-slate-100 cursor-pointer transition-all flex gap-3 group/item relative ${
                              selectedPaperId === paper.id ? 'bg-blue-50/70 border-l-4 border-l-vnu-blue ring-1 ring-inset ring-vnu-blue/10' : 'hover:bg-slate-50 border-l-4 border-l-transparent'
                           }`}
                        >
                           <div className={`w-2.5 h-2.5 rounded-full mt-1 shrink-0 ring-4 ${
                              paper.status === PaperStatus.ABSTRACT_INCLUDE ? 'bg-emerald-500 ring-emerald-50' :
                              paper.status === PaperStatus.ABSTRACT_EXCLUDE ? 'bg-rose-500 ring-rose-50' :
                              paper.status === PaperStatus.MAYBE ? 'bg-amber-500 ring-amber-50' : 'bg-slate-300 ring-slate-50'
                           }`}></div>
                           <div className="flex-1 pr-8">
                              <h4 className={`text-[13px] font-bold line-clamp-2 leading-snug mb-2 transition-colors ${
                                 selectedPaperId === paper.id ? 'text-vnu-blue' : 'text-slate-800 group-hover/item:text-vnu-blue'
                              }`}>{paper.title}</h4>
                              <div className="flex items-center justify-between text-[10px] text-slate-500 font-bold uppercase tracking-wider">
                                 <span className="truncate max-w-[150px]">{paper.authors?.split(',')[0]}</span>
                                 <span className="bg-slate-100 px-2 py-0.5 rounded-md">{paper.year}</span>
                              </div>
                              {paper.aiScreeningReason && (
                                <div className="mt-2 text-[10px] font-semibold text-blue-600 bg-blue-50 inline-block px-1.5 py-0.5 rounded">
                                  ⚡ AI Scored
                                </div>
                              )}
                           </div>
                           {/* Delete button — appears on hover */}
                           <button
                             onClick={(e) => handleRemovePaper(paper.id, e)}
                             title="Remove paper"
                             className="absolute top-3 right-3 opacity-0 group-hover/item:opacity-100 transition-all p-1.5 rounded-lg bg-white border border-slate-200 text-slate-400 hover:bg-rose-500 hover:text-white hover:border-rose-500 shadow-sm"
                           >
                             <svg xmlns="http://www.w3.org/2000/svg" className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                               <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                             </svg>
                           </button>
                        </div>
                     ))}
                     {paginatedPapers.length === 0 && (
                        <div className="flex-1 flex flex-col items-center justify-center p-8 text-center bg-slate-50/50">
                           <svg xmlns="http://www.w3.org/2000/svg" className="h-12 w-12 text-slate-300 mb-3" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" /></svg>
                           <p className="text-slate-400 font-bold text-sm">No results in this stage.</p>
                        </div>
                     )}
                  </div>

                  {/* Right Column: Split Detail View */}
                  <div className="w-[62%] bg-white border border-slate-200 rounded-2xl shadow-sm flex flex-col overflow-hidden relative">
                     {(() => {
                        const paper = filteredPapers.find(p => p.id === selectedPaperId);
                        if (!paper) return (
                           <div className="flex-1 flex flex-col items-center justify-center text-slate-400 bg-slate-50/20">
                              <svg xmlns="http://www.w3.org/2000/svg" className="h-16 w-16 mb-4 text-slate-200" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M15 15l-2 5L9 9l11 4-5 2zm0 0l5 5M7.188 2.239l.777 2.897M5.136 7.965l-2.898-.777M13.95 4.05l-2.122 2.122m-5.657 5.656l-2.12 2.122" /></svg>
                              <span className="font-bold">Select an article from the list to screen</span>
                           </div>
                        );
                        return (
                           <>
                              {/* Action Bar (Sticky) */}
                              <div className="sticky top-0 bg-white/95 backdrop-blur-md border-b border-slate-200 p-4 z-10 flex gap-3 justify-center items-stretch">
                                 <button 
                                    onClick={() => updatePaperStatus(paper.id, PaperStatus.ABSTRACT_INCLUDE)}
                                    className={`relative flex flex-col items-center justify-center w-full max-w-[130px] px-2 py-3 rounded-xl transition-all border ${
                                       paper.status === PaperStatus.ABSTRACT_INCLUDE ? 'bg-emerald-600 border-emerald-600 text-white shadow-xl shadow-emerald-200 scale-[1.02]' : 'bg-white border-slate-200 text-slate-600 hover:border-emerald-500 hover:text-emerald-600'
                                    }`}
                                 >
                                    <span className="font-black uppercase tracking-widest text-sm mb-1 z-10 relative">Include</span>
                                    <span className={`text-[10px] font-bold z-10 relative ${paper.status === PaperStatus.ABSTRACT_INCLUDE ? 'text-emerald-200' : 'text-slate-400'}`}>Shortcut: I</span>
                                    {paper.status === PaperStatus.ABSTRACT_INCLUDE && <div className="absolute inset-0 bg-white/20 blur-sm rounded-xl"></div>}
                                 </button>
                                 
                                 <button 
                                    onClick={() => updatePaperStatus(paper.id, PaperStatus.MAYBE)}
                                    className={`relative flex flex-col items-center justify-center w-full max-w-[130px] px-2 py-3 rounded-xl transition-all border ${
                                       paper.status === PaperStatus.MAYBE ? 'bg-amber-500 border-amber-500 text-white shadow-xl shadow-amber-200 scale-[1.02]' : 'bg-white border-slate-200 text-slate-600 hover:border-amber-500 hover:text-amber-600'
                                    }`}
                                 >
                                    <span className="font-black uppercase tracking-widest text-sm mb-1 z-10 relative">Maybe</span>
                                    <span className={`text-[10px] font-bold z-10 relative ${paper.status === PaperStatus.MAYBE ? 'text-amber-100' : 'text-slate-400'}`}>Shortcut: M</span>
                                 </button>

                                 <button 
                                    onClick={() => updatePaperStatus(paper.id, PaperStatus.ABSTRACT_EXCLUDE)}
                                    className={`relative flex flex-col items-center justify-center w-full max-w-[130px] px-2 py-3 rounded-xl transition-all border ${
                                       paper.status === PaperStatus.ABSTRACT_EXCLUDE ? 'bg-rose-600 border-rose-600 text-white shadow-xl shadow-rose-200 scale-[1.02]' : 'bg-white border-slate-200 text-slate-600 hover:border-rose-500 hover:text-rose-600'
                                    }`}
                                 >
                                    <span className="font-black uppercase tracking-widest text-sm mb-1 z-10 relative">Exclude</span>
                                    <span className={`text-[10px] font-bold z-10 relative ${paper.status === PaperStatus.ABSTRACT_EXCLUDE ? 'text-rose-200' : 'text-slate-400'}`}>Shortcut: E</span>
                                 </button>

                                 {/* Delete button in detail panel */}
                                 <button
                                    onClick={(e) => handleRemovePaper(paper.id, e)}
                                    title="Remove this paper from the review"
                                    className="flex flex-col items-center justify-center px-4 py-3 rounded-xl border border-slate-200 text-slate-400 hover:bg-rose-600 hover:text-white hover:border-rose-600 hover:shadow-xl hover:shadow-rose-200 transition-all shrink-0"
                                 >
                                    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mb-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                                    </svg>
                                    <span className="text-[10px] font-black uppercase tracking-widest">Delete</span>
                                 </button>
                              </div>

                              {/* Document Detail Body */}
                              <div className="p-8 overflow-y-auto flex-1 hide-scrollbar">
                                 <h2 className="text-[22px] font-bold text-slate-900 mb-6 leading-relaxed font-serif">{paper.title}</h2>
                                 
                                 <div className="flex flex-wrap items-center gap-3 mb-8">
                                    <span className="bg-slate-100 text-slate-600 px-3 py-1.5 rounded-lg text-xs font-bold uppercase tracking-wider">{paper.year}</span>
                                    <div className="w-1 h-1 bg-slate-300 rounded-full"></div>
                                    <span className="text-slate-600 text-sm font-semibold">{paper.authors}</span>
                                    <div className="w-1 h-1 bg-slate-300 rounded-full"></div>
                                    <span className="text-vnu-blue text-sm font-semibold italic">{paper.journal}</span>
                                 </div>

                                 {paper.aiScreeningReason && (
                                     <div className="bg-blue-50/50 border border-blue-100 p-5 rounded-2xl mb-8">
                                         <div className="flex items-start gap-4">
                                             <div className="bg-white p-2.5 rounded-xl border border-blue-100 text-blue-600 shadow-sm shrink-0 mt-1">
                                                <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" viewBox="0 0 20 20" fill="currentColor">
                                                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                                                </svg>
                                             </div>
                                             <div>
                                                 <h4 className="flex items-center gap-2 text-blue-900 font-bold text-sm uppercase tracking-widest mb-2">
                                                   AI Evaluation Rationale
                                                   <span className="bg-blue-600 text-white text-[9px] px-2 py-0.5 rounded-full">Beta</span>
                                                 </h4>
                                                 <p className="text-blue-900/80 text-[15px] leading-relaxed font-medium">{paper.aiScreeningReason}</p>
                                             </div>
                                         </div>
                                     </div>
                                 )}

                                 <div className="prose max-w-none">
                                     <h3 className="flex items-center gap-2 text-xs font-black uppercase tracking-[0.2em] text-slate-400 mb-6">
                                       <span className="w-4 h-[2px] bg-slate-300"></span> Abstract
                                     </h3>
                                     <p className="text-slate-700 text-[16px] leading-relaxed whitespace-pre-line text-justify font-serif">{paper.abstract}</p>
                                 </div>

                                 {paper.extractionData && (
                                    <div className="mt-10 pt-8 border-t border-slate-100">
                                       <h3 className="flex items-center gap-2 text-xs font-black uppercase tracking-[0.2em] text-emerald-600 mb-6">
                                         <span className="w-4 h-[2px] bg-emerald-500"></span> Extracted Structured Data (QA)
                                       </h3>
                                       <div className="bg-white border border-emerald-100 shadow-lg shadow-emerald-100/50 rounded-2xl overflow-hidden text-sm">
                                          <div className="grid grid-cols-[140px_1fr] border-b border-emerald-50">
                                             <div className="bg-emerald-50/50 p-4 font-bold text-emerald-800 uppercase tracking-widest text-[11px] flex items-center">Methodology</div>
                                             <div className="p-4 text-slate-700 font-medium">{paper.extractionData.methodology}</div>
                                          </div>
                                          <div className="grid grid-cols-[140px_1fr] border-b border-emerald-50">
                                             <div className="bg-emerald-50/50 p-4 font-bold text-emerald-800 uppercase tracking-widest text-[11px] flex items-center">Sample Size</div>
                                             <div className="p-4 text-slate-700 font-medium">{paper.extractionData.sampleSize}</div>
                                          </div>
                                          <div className="grid grid-cols-[140px_1fr] border-b border-emerald-50">
                                             <div className="bg-emerald-50/50 p-4 font-bold text-emerald-800 uppercase tracking-widest text-[11px] flex items-center">Key Findings</div>
                                             <div className="p-4 text-slate-700 font-medium">{paper.extractionData.keyFindings}</div>
                                          </div>
                                          <div className="grid grid-cols-[140px_1fr] border-b border-emerald-50">
                                             <div className="bg-emerald-50/50 p-4 font-bold text-emerald-800 uppercase tracking-widest text-[11px] flex items-center">Limitations</div>
                                             <div className="p-4 text-slate-700 font-medium">{paper.extractionData.limitations}</div>
                                          </div>
                                          <div className="grid grid-cols-[140px_1fr]">
                                             <div className="bg-rose-50/50 p-4 font-bold text-rose-800 uppercase tracking-widest text-[11px] flex items-center">Risk of Bias</div>
                                             <div className="p-4 text-slate-700 font-medium">{paper.extractionData.riskOfBias}</div>
                                          </div>
                                       </div>
                                    </div>
                                 )}
                              </div>
                           </>
                        );
                     })()}
                  </div>
                </div>
              </>
            )}
          </div>
        </main>
      </div>
    </div>
  );
};

export default App;
