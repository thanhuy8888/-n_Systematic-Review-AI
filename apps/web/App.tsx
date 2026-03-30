
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

                <div className="space-y-6">
                  {paginatedPapers.map(paper => (
                    <div key={paper.id} className="group bg-white border border-slate-200 rounded-2xl overflow-hidden shadow-sm hover:shadow-xl hover:border-vnu-blue/30 transition-all flex">
                      <div className={`w-2 ${
                        paper.status === PaperStatus.ABSTRACT_INCLUDE ? 'bg-emerald-500' :
                        paper.status === PaperStatus.ABSTRACT_EXCLUDE ? 'bg-rose-500' :
                        paper.status === PaperStatus.MAYBE ? 'bg-amber-500' : 'bg-slate-200'
                      }`}></div>
                      
                      <div className="flex-1 p-6">
                        <div className="flex justify-between items-start gap-6 mb-4">
                          <div className="flex-1">
                            <h3 className="font-bold text-slate-900 text-lg leading-snug mb-2 group-hover:text-vnu-blue transition-colors">{paper.title}</h3>
                            <div className="flex flex-wrap items-center gap-y-2 gap-x-4 text-[11px] font-bold text-slate-400 uppercase tracking-wider">
                              <div className="flex items-center gap-1.5">
                                <div className="w-5 h-5 rounded-full bg-slate-100 flex items-center justify-center text-slate-500">
                                  <svg xmlns="http://www.w3.org/2000/svg" className="h-3 w-3" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" /></svg>
                                </div>
                                <span>{paper.authors?.split(',')[0]} et al.</span>
                              </div>
                              <div className="flex items-center gap-1.5">
                                <div className="w-5 h-5 rounded-full bg-blue-50 flex items-center justify-center text-vnu-blue">
                                  <svg xmlns="http://www.w3.org/2000/svg" className="h-3 w-3" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" /></svg>
                                </div>
                                <span className="text-vnu-blue">{paper.year}</span>
                              </div>
                              <div className="flex items-center gap-1.5">
                                <div className="w-5 h-5 rounded-full bg-slate-100 flex items-center justify-center text-slate-500">
                                  <svg xmlns="http://www.w3.org/2000/svg" className="h-3 w-3" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" /></svg>
                                </div>
                                <span className="italic normal-case font-medium">{paper.journal}</span>
                              </div>
                            </div>
                          </div>
                          
                          <div className="flex flex-row md:flex-col gap-2 shrink-0">
                            <button 
                              onClick={() => updatePaperStatus(paper.id, PaperStatus.ABSTRACT_INCLUDE)}
                              className={`px-4 py-2 text-[10px] font-black uppercase rounded-xl transition-all shadow-sm ${
                                paper.status === PaperStatus.ABSTRACT_INCLUDE ? 'bg-emerald-600 text-white shadow-emerald-200' : 'bg-emerald-50 text-emerald-600 hover:bg-emerald-100'
                              }`}
                            >
                              Include
                            </button>
                            <button 
                              onClick={() => updatePaperStatus(paper.id, PaperStatus.MAYBE)}
                              className={`px-4 py-2 text-[10px] font-black uppercase rounded-xl transition-all shadow-sm ${
                                paper.status === PaperStatus.MAYBE ? 'bg-amber-500 text-white shadow-amber-200' : 'bg-amber-50 text-amber-600 hover:bg-amber-100'
                              }`}
                            >
                              Maybe
                            </button>
                            <button 
                              onClick={() => updatePaperStatus(paper.id, PaperStatus.ABSTRACT_EXCLUDE)}
                              className={`px-4 py-2 text-[10px] font-black uppercase rounded-xl transition-all shadow-sm ${
                                paper.status === PaperStatus.ABSTRACT_EXCLUDE ? 'bg-rose-600 text-white shadow-rose-200' : 'bg-rose-50 text-rose-600 hover:bg-rose-100'
                              }`}
                            >
                              Exclude
                            </button>
                            <button 
                              onClick={(e) => handleRemovePaper(paper.id, e)}
                              className="px-4 py-2 text-[10px] font-black uppercase rounded-xl transition-all shadow-sm bg-red-50 text-red-600 hover:bg-red-100 flex items-center justify-center"
                              title="Delete Paper"
                            >
                              <svg xmlns="http://www.w3.org/2000/svg" className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" /></svg>
                            </button>
                          </div>
                        </div>

                        <div className="relative mb-5">
                          <p className="text-sm text-slate-600 leading-relaxed line-clamp-3 font-medium">
                            {paper.abstract}
                          </p>
                          <div className="absolute bottom-0 left-0 w-full h-6 bg-gradient-to-t from-white to-transparent"></div>
                        </div>

                        {paper.aiScreeningReason && (
                          <div className="bg-vnu-blue/[0.03] border border-vnu-blue/10 p-4 rounded-2xl text-xs text-slate-600 mb-5 relative overflow-hidden">
                            <div className="absolute top-0 left-0 w-1 h-full bg-vnu-blue/40"></div>
                            <div className="flex items-start gap-3">
                              <div className="bg-vnu-blue/10 p-1.5 rounded-lg text-vnu-blue">
                                <svg xmlns="http://www.w3.org/2000/svg" className="h-3.5 w-3.5" viewBox="0 0 20 20" fill="currentColor">
                                  <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
                                </svg>
                              </div>
                              <p className="leading-relaxed">
                                <span className="font-black text-vnu-blue mr-2 uppercase tracking-tighter">AI Verdict:</span>
                                {paper.aiScreeningReason}
                              </p>
                            </div>
                          </div>
                        )}

                        {paper.extractionData && (
                          <div className="bg-emerald-50 border border-emerald-100 p-4 rounded-2xl text-xs text-slate-700 mb-5">
                             <h4 className="font-bold mb-2 text-emerald-800 uppercase tracking-widest text-[10px]">Extracted Data</h4>
                             <p className="mb-1"><span className="font-bold text-emerald-600">Methodology:</span> {paper.extractionData.methodology}</p>
                             <p className="mb-1"><span className="font-bold text-emerald-600">Sample Size:</span> {paper.extractionData.sampleSize}</p>
                             <p className="mb-1"><span className="font-bold text-emerald-600">Key Findings:</span> {paper.extractionData.keyFindings}</p>
                             <p className="mb-1"><span className="font-bold text-emerald-600">Limitations:</span> {paper.extractionData.limitations}</p>
                             <p><span className="font-bold text-emerald-600">Bias Risk:</span> {paper.extractionData.riskOfBias}</p>
                          </div>
                        )}

                        <div className="flex items-center gap-2">
                          {paper.keywords?.slice(0, 6).map(kw => (
                            <span key={kw} className="text-[9px] bg-slate-100 text-slate-500 px-2.5 py-1 rounded-lg font-bold uppercase tracking-tighter border border-slate-200/50">
                              {kw}
                            </span>
                          ))}
                        </div>
                      </div>
                    </div>
                  ))}
                  
                  {paginatedPapers.length === 0 && (
                    <div className="py-20 text-center bg-white rounded-2xl border border-dashed border-slate-300">
                      <p className="text-slate-400 font-bold">No results found in this stage.</p>
                    </div>
                  )}
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
