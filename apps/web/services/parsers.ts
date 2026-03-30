
import { Paper, PaperStatus } from "../types";

export const parseRIS = (content: string, fileName: string): Paper[] => {
  const papers: Paper[] = [];
  const entries = content.split(/ER  - /);

  entries.forEach(entry => {
    if (!entry.trim()) return;
    
    const titleMatch = entry.match(/^TI  - (.*)$/m) || entry.match(/^T1  - (.*)$/m);
    const abstractMatch = entry.match(/^AB  - ([\s\S]*?)(?=\n[A-Z]{2}  -|$)/m);
    const authorMatch = entry.match(/^AU  - (.*)$/gm);
    const yearMatch = entry.match(/^PY  - (.*)$/m);
    const journalMatch = entry.match(/^JO  - (.*)$/m) || entry.match(/^JF  - (.*)$/m);
    const doiMatch = entry.match(/^DO  - (.*)$/m);
    const keywordMatch = entry.match(/^KW  - (.*)$/gm); // Extract keywords

    if (titleMatch) {
      papers.push({
        id: Math.random().toString(36).substr(2, 9),
        title: titleMatch[1].trim(),
        abstract: abstractMatch ? abstractMatch[1].trim() : "No abstract available",
        authors: authorMatch ? authorMatch.map(a => a.replace('AU  - ', '').trim()).join(', ') : "",
        year: yearMatch ? yearMatch[1].trim() : "",
        journal: journalMatch ? journalMatch[1].trim() : "",
        doi: doiMatch ? doiMatch[1].trim() : "",
        keywords: keywordMatch ? keywordMatch.map(k => k.replace('KW  - ', '').trim()) : [],
        status: PaperStatus.PENDING,
        sourceFile: fileName
      });
    }
  });

  return papers;
};

export const parseBIB = (content: string, fileName: string): Paper[] => {
  const papers: Paper[] = [];
  const entries = content.split(/@\w+\{/);

  entries.forEach(entry => {
    if (!entry.trim()) return;
    
    const titleMatch = entry.match(/title\s*=\s*[\{"]([\s\S]*?)[\}"]/i);
    const abstractMatch = entry.match(/abstract\s*=\s*[\{"]([\s\S]*?)[\}"]/i);
    const authorMatch = entry.match(/author\s*=\s*[\{"]([\s\S]*?)[\}"]/i);
    const yearMatch = entry.match(/year\s*=\s*[\{"](\d{4})[\}"]/i);
    const journalMatch = entry.match(/journal\s*=\s*[\{"]([\s\S]*?)[\}"]/i);
    const keywordsMatch = entry.match(/keywords\s*=\s*[\{"]([\s\S]*?)[\}"]/i);

    if (titleMatch) {
      papers.push({
        id: Math.random().toString(36).substr(2, 9),
        title: titleMatch[1].replace(/[\{\}]/g, '').trim(),
        abstract: abstractMatch ? abstractMatch[1].replace(/[\{\}]/g, '').trim() : "No abstract available",
        authors: authorMatch ? authorMatch[1].replace(/[\{\}]/g, '').trim() : "",
        year: yearMatch ? yearMatch[1].trim() : "",
        journal: journalMatch ? journalMatch[1].replace(/[\{\}]/g, '').trim() : "",
        keywords: keywordsMatch ? keywordsMatch[1].replace(/[\{\}]/g, '').split(',').map(k => k.trim()) : [],
        status: PaperStatus.PENDING,
        sourceFile: fileName
      });
    }
  });

  return papers;
};
