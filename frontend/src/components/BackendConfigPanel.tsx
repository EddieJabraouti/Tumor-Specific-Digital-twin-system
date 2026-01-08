import React, { useState, useEffect } from 'react';
import { setApiConfig, getApiConfig } from '../api/endpoints';
import styles from './BackendConfigPanel.module.css';

const EXPECTED_ENDPOINTS = [
  {
    method: 'POST',
    path: '/api/preprocess',
    description: 'Start preprocessing job',
    payload: '{ modalities: { t1, t1ce, t2, flair }, mask?: File }',
    response: '{ jobId: string }',
  },
  {
    method: 'POST',
    path: '/api/segment',
    description: 'Start segmentation job',
    payload: '{ modalities: { t1, t1ce, t2, flair }, mask?: File }',
    response: '{ jobId: string }',
  },
  {
    method: 'POST',
    path: '/api/diffuse',
    description: 'Start diffusion generation job',
    payload: '{ modalities: { t1, t1ce, t2, flair }, segmentationMask: File }',
    response: '{ jobId: string }',
  },
  {
    method: 'POST',
    path: '/api/tumor3d',
    description: 'Start 3D modeling job',
    payload: '{ segmentationMask: File }',
    response: '{ jobId: string }',
  },
  {
    method: 'POST',
    path: '/api/simulate',
    description: 'Start simulation job',
    payload: '{ tumorModel: File, segmentationMask?: File, parameters?: object }',
    response: '{ jobId: string }',
  },
  {
    method: 'GET',
    path: '/api/jobs/:jobId',
    description: 'Get job status',
    response: '{ status: "queued"|"running"|"succeeded"|"failed", progress: number, result?: object, error?: string }',
  },
];

export const BackendConfigPanel: React.FC = () => {
  const config = getApiConfig();
  const [baseUrl, setBaseUrl] = useState(config.baseUrl);
  const [mockMode, setMockMode] = useState(config.mockMode);

  useEffect(() => {
    setApiConfig({ baseUrl, mockMode });
  }, [baseUrl, mockMode]);

  return (
    <div className={styles.panel}>
      <h2>Backend Configuration</h2>

      <div className={styles.configSection}>
        <div className={styles.configItem}>
          <label>
            Base URL:
            <input
              type="text"
              value={baseUrl}
              onChange={(e) => setBaseUrl(e.target.value)}
              placeholder="http://localhost:8000"
              disabled={mockMode}
            />
          </label>
        </div>

        <div className={styles.configItem}>
          <label>
            <input
              type="checkbox"
              checked={mockMode}
              onChange={(e) => setMockMode(e.target.checked)}
            />
            Mock Mode (ON: uses local mock functions, OFF: uses real API calls)
          </label>
        </div>
      </div>

      <div className={styles.endpointsSection}>
        <h3>Expected API Endpoints</h3>
        <p className={styles.description}>
          The backend should implement these endpoints with the specified request/response formats:
        </p>
        <div className={styles.endpointsList}>
          {EXPECTED_ENDPOINTS.map((endpoint, idx) => (
            <div key={idx} className={styles.endpoint}>
              <div className={styles.endpointHeader}>
                <span className={styles.method}>{endpoint.method}</span>
                <span className={styles.path}>{endpoint.path}</span>
              </div>
              <div className={styles.endpointDescription}>{endpoint.description}</div>
              {endpoint.payload && (
                <div className={styles.endpointDetail}>
                  <strong>Payload:</strong> <code>{endpoint.payload}</code>
                </div>
              )}
              <div className={styles.endpointDetail}>
                <strong>Response:</strong> <code>{endpoint.response}</code>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className={styles.jobSchemaSection}>
        <h3>Job Status Schema</h3>
        <pre className={styles.schema}>
{`{
  "status": "queued" | "running" | "succeeded" | "failed",
  "progress": number,  // 0-100
  "result"?: {
    "artifacts": Array<{
      "name": string,
      "url": string,
      "kind": "nifti" | "json" | "mesh" | "image" | "other"
    }>,
    "metadata"?: object,
    // Step-specific fields:
    // - tumor3d: { "modelUrl": string, "format": "glb"|"obj", "stats"?: {...} }
    // - simulate: { "frames": Array<{ "t": number, "artifactUrl": string }> }
  },
  "error"?: string
}`}
        </pre>
      </div>
    </div>
  );
};
