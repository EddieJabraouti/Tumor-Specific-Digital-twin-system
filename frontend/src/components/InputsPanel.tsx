import React from 'react';
import { usePipelineStore } from '../state/pipelineStore';
import styles from './InputsPanel.module.css';

const MODALITIES = [
  { key: 't1' as const, label: 'T1' },
  { key: 't1ce' as const, label: 'T1ce' },
  { key: 't2' as const, label: 'T2' },
  { key: 'flair' as const, label: 'FLAIR' },
] as const;

export const InputsPanel: React.FC = () => {
  const uploadedFiles = usePipelineStore((state) => state.uploadedFiles);
  const setUploadedFile = usePipelineStore((state) => state.setUploadedFile);

  const handleFileChange = (modality: keyof typeof uploadedFiles, event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    setUploadedFile(modality, file);
  };

  const uploadedCount = MODALITIES.filter((m) => uploadedFiles[m.key]).length;
  const hasMask = !!uploadedFiles.mask;

  return (
    <div className={styles.panel}>
      <h2>Inputs</h2>
      
      <div className={styles.section}>
        <h3>MRI Modalities</h3>
        <div className={styles.fileInputs}>
          {MODALITIES.map((modality) => (
            <div key={modality.key} className={styles.fileInput}>
              <label>
                {modality.label}
                <input
                  type="file"
                  accept=".nii,.nii.gz"
                  onChange={(e) => handleFileChange(modality.key, e)}
                />
                {uploadedFiles[modality.key] && (
                  <span className={styles.fileName}>
                    {uploadedFiles[modality.key]?.name}
                  </span>
                )}
              </label>
            </div>
          ))}
        </div>
      </div>

      <div className={styles.section}>
        <h3>Optional: Segmentation Mask</h3>
        <div className={styles.fileInput}>
          <label>
            <input
              type="checkbox"
              checked={!!uploadedFiles.mask}
              onChange={(e) => {
                if (!e.target.checked) {
                  setUploadedFile('mask', undefined);
                }
              }}
            />
            Upload existing segmentation mask
          </label>
          {uploadedFiles.mask && (
            <input
              type="file"
              accept=".nii,.nii.gz"
              onChange={(e) => handleFileChange('mask', e)}
            />
          )}
          {uploadedFiles.mask && (
            <span className={styles.fileName}>
              {uploadedFiles.mask.name}
            </span>
          )}
        </div>
      </div>

      <div className={styles.checklist}>
        <h3>Input Completeness</h3>
        <div className={styles.checklistItem}>
          <span className={uploadedCount === 4 ? styles.check : styles.uncheck}>
            {uploadedCount === 4 ? '✓' : '○'}
          </span>
          <span>4 modalities uploaded ({uploadedCount}/4)</span>
        </div>
        <div className={styles.checklistItem}>
          <span className={hasMask ? styles.check : styles.uncheck}>
            {hasMask ? '✓' : '○'}
          </span>
          <span>Segmentation mask (optional)</span>
        </div>
      </div>
    </div>
  );
};
