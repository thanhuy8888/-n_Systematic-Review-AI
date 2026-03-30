
import { GoogleGenAI, Type } from "@google/genai";
import { Paper, ReviewCriteria, PaperStatus } from "../types";

// Always use const ai = new GoogleGenAI({apiKey: process.env.API_KEY});
const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

export const screenPaper = async (paper: Paper, criteria: ReviewCriteria): Promise<{ status: PaperStatus; reason: string }> => {
  const prompt = `
    You are an expert researcher performing a systematic review. 
    Evaluate the following paper based on the specified PICO/Inclusion criteria.
    
    CRITERIA:
    - Population: ${criteria.population}
    - Intervention: ${criteria.intervention}
    - Comparison: ${criteria.comparison}
    - Outcome: ${criteria.outcome}
    - Study Type: ${criteria.studyType}
    
    PAPER:
    - Title: ${paper.title}
    - Abstract: ${paper.abstract}
    
    Respond in JSON format with two fields:
    1. "decision": either "INCLUDE" or "EXCLUDE"
    2. "reason": a short explanation (1-2 sentences) of why it was included or excluded.
  `;

  try {
    const response = await ai.models.generateContent({
      model: 'gemini-3-flash-preview',
      contents: prompt,
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            decision: { type: Type.STRING, description: "INCLUDE or EXCLUDE" },
            reason: { type: Type.STRING }
          },
          required: ["decision", "reason"]
        }
      }
    });

    const result = JSON.parse(response.text);
    // Fix: Use correct PaperStatus enum values instead of non-existent SCREENED_INCLUDE/EXCLUDE
    return {
      status: result.decision === "INCLUDE" ? PaperStatus.ABSTRACT_INCLUDE : PaperStatus.ABSTRACT_EXCLUDE,
      reason: result.reason
    };
  } catch (error) {
    console.error("Screening failed", error);
    return { status: PaperStatus.PENDING, reason: "Error processing paper" };
  }
};

export const extractPaperData = async (paper: Paper) => {
  const prompt = `
    Extract detailed study data from the following paper content for a systematic review synthesis.
    
    PAPER TITLE: ${paper.title}
    CONTENT: ${paper.abstract} ${paper.fullText || ''}
    
    Focus on extracting:
    - Methodology
    - Sample Size
    - Key Findings
    - Limitations
    - Risk of Bias (low/medium/high and why)
  `;

  try {
    const response = await ai.models.generateContent({
      model: 'gemini-3-pro-preview',
      contents: prompt,
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            methodology: { type: Type.STRING },
            sampleSize: { type: Type.STRING },
            keyFindings: { type: Type.STRING },
            limitations: { type: Type.STRING },
            riskOfBias: { type: Type.STRING }
          },
          required: ["methodology", "sampleSize", "keyFindings", "limitations", "riskOfBias"]
        }
      }
    });

    return JSON.parse(response.text);
  } catch (error) {
    console.error("Extraction failed", error);
    return null;
  }
};
