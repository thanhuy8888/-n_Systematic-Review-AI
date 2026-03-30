
import { Paper, ReviewCriteria } from "../types";

const DB_NAME = "SystematicReviewDB";
const STORE_NAME = "projects";
const DB_VERSION = 1;

export const openDB = (): Promise<IDBDatabase> => {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME);
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
};

export const saveProject = async (papers: Paper[], criteria: ReviewCriteria) => {
  const db = await openDB();
  const tx = db.transaction(STORE_NAME, "readwrite");
  const store = tx.objectStore(STORE_NAME);
  store.put({ papers, criteria, lastSaved: new Date().toISOString() }, "current_project");
  return tx.oncomplete;
};

export const loadProject = async (): Promise<{ papers: Paper[]; criteria: ReviewCriteria } | null> => {
  const db = await openDB();
  return new Promise((resolve) => {
    const tx = db.transaction(STORE_NAME, "readonly");
    const store = tx.objectStore(STORE_NAME);
    const request = store.get("current_project");
    request.onsuccess = () => resolve(request.result || null);
    request.onerror = () => resolve(null);
  });
};

export const exportProjectFile = (papers: Paper[], criteria: ReviewCriteria) => {
  const data = JSON.stringify({ papers, criteria, version: "1.0", exportDate: new Date().toISOString() });
  const blob = new Blob([data], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `SR_Project_${new Date().toISOString().split('T')[0]}.json`;
  a.click();
  URL.revokeObjectURL(url);
};
