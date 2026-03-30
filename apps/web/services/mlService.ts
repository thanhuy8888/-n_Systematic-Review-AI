import { Paper, ReviewCriteria, PaperStatus } from "../types";

export const screenPaper = async (paper: Paper, criteria: ReviewCriteria): Promise<{ status: PaperStatus; reason: string }> => {
  try {
    const response = await fetch('http://localhost:8000/screening/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ paper, criteria })
    });
    
    if (!response.ok) throw new Error("API request failed");
    
    const result = await response.json();
    return {
      status: result.decision === "INCLUDE" ? PaperStatus.ABSTRACT_INCLUDE : PaperStatus.ABSTRACT_EXCLUDE,
      reason: result.reason
    };
  } catch (error) {
    console.error("Screening failed", error);
    return { status: PaperStatus.PENDING, reason: "Error processing paper with local backend." };
  }
};

export const extractPaperData = async (paper: Paper) => {
  try {
    const response = await fetch('http://localhost:8000/screening/extract', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ paper })
    });
    
    if (!response.ok) throw new Error("API request failed");
    
    return await response.json();
  } catch (error) {
    console.error("Extraction failed", error);
    return null;
  }
};
