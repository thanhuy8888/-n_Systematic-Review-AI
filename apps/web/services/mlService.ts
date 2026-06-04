import { Paper, ReviewCriteria, PaperStatus } from "../types";

/**
 * API base URL.
 * - In Docker production: empty string → nginx proxies /screening/* to backend
 * - In local dev: http://localhost:8000 (set VITE_API_URL in .env.local)
 */
const API_BASE = (import.meta as any).env?.VITE_API_URL ?? "";

export const screenPaper = async (
  paper: Paper,
  criteria: ReviewCriteria
): Promise<{ status: PaperStatus; reason: string }> => {
  try {
    const response = await fetch(`${API_BASE}/screening/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ paper, criteria }),
    });

    if (!response.ok) throw new Error(`API ${response.status}`);

    const result = await response.json();
    return {
      status:
        result.decision === "INCLUDE"
          ? PaperStatus.ABSTRACT_INCLUDE
          : PaperStatus.ABSTRACT_EXCLUDE,
      reason: result.reason,
    };
  } catch (error) {
    console.error("Screening failed", error);
    return {
      status: PaperStatus.PENDING,
      reason: "Error connecting to AI backend.",
    };
  }
};

export const extractPaperData = async (paper: Paper) => {
  try {
    const response = await fetch(`${API_BASE}/screening/extract`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ paper }),
    });

    if (!response.ok) throw new Error(`API ${response.status}`);

    return await response.json();
  } catch (error) {
    console.error("Extraction failed", error);
    return null;
  }
};
