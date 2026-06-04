
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
  exclusionReason?: string;
}

/** Evidence table fields for murine metabolic disorder studies (PICO-aligned) */
export interface ExtractionData {
  // Study design
  methodology: string;
  sampleSize: string;
  // Population (mouse model)
  mouseStrain: string;
  sexAge: string;
  // Intervention
  dietType: string;
  dietComposition: string;
  duration: string;
  // Comparator
  controlDiet: string;
  // Outcomes — lipid profile
  tcTg: string;
  ldlHdl: string;
  // Outcomes — glucose/insulin axis
  glucoseInsulin: string;
  homaIr: string;
  // Outcomes — liver
  altAst: string;
  liverHistology: string;
  // Summary
  keyFindings: string;
  limitations: string;
  riskOfBias: string;
  // Evidence traceability: raw text span supporting the finding
  evidenceSpan?: string;
}

export interface ReviewCriteria {
  population: string;
  intervention: string;
  comparison: string;
  outcome: string;
  studyType: string;
}
