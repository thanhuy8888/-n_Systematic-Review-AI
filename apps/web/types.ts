
export enum PaperStatus {
  PENDING = 'PENDING',
  ABSTRACT_INCLUDE = 'ABSTRACT_INCLUDE',
  ABSTRACT_EXCLUDE = 'ABSTRACT_EXCLUDE',
  FULLTEXT_INCLUDE = 'FULLTEXT_INCLUDE',
  FULLTEXT_EXCLUDE = 'FULLTEXT_EXCLUDE',
  EXTRACTED = 'EXTRACTED',
  MAYBE = 'MAYBE'
}

export enum ReviewStage {
  IDENTIFICATION = 'Identification',
  ABSTRACT_SCREENING = 'Abstract Screening',
  FULLTEXT_SCREENING = 'Full-text Screening',
  EXTRACTION = 'Extraction'
}

export interface Paper {
  id: string;
  title: string;
  abstract: string;
  fullText?: string;
  authors?: string;
  year?: string;
  journal?: string;
  doi?: string;
  keywords?: string[];
  status: PaperStatus;
  aiScreeningReason?: string;
  extractionData?: ExtractionData;
  sourceFile?: string;
}

export interface ExtractionData {
  methodology: string;
  sampleSize: string;
  keyFindings: string;
  limitations: string;
  riskOfBias: string;
}

export interface ReviewCriteria {
  population: string;
  intervention: string;
  comparison: string;
  outcome: string;
  studyType: string;
}
