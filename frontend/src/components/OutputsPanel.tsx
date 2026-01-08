import React, { useState } from 'react';
import { usePipelineStore } from '../state/pipelineStore';
import styles from './OutputsPanel.module.css';

export const OutputsPanel: React.FC = () => {
  const outputs = usePipelineStore((state) => state.outputs);
  const steps = usePipelineStore((state) => state.steps);
  const [activeTab, setActiveTab] = useState<'segmentation' | 'synthetic' | '3d' | 'simulation'>('segmentation');

  const segmentationOutput = outputs.segment || outputs['re-segment'];
  const syntheticOutput = outputs.diffuse;
  const modelOutput = outputs.tumor3d;
  const simulationOutput = outputs.simulate;

  const renderArtifacts = (artifacts: any[] | undefined) => {
    if (!artifacts || artifacts.length === 0) {
      return <div className={styles.empty}>No artifacts available</div>;
    }

    return (
      <div className={styles.artifacts}>
        {artifacts.map((artifact, idx) => (
          <div key={idx} className={styles.artifact}>
            <div className={styles.artifactName}>{artifact.name}</div>
            <div className={styles.artifactUrl}>
              {artifact.url ? (
                <a href={artifact.url} target="_blank" rel="noopener noreferrer">
                  {artifact.url}
                </a>
              ) : (
                <span>No URL available</span>
              )}
            </div>
            <div className={styles.artifactKind}>Type: {artifact.kind}</div>
          </div>
        ))}
      </div>
    );
  };

  return (
    <div className={styles.panel}>
      <h2>Outputs</h2>

      <div className={styles.tabs}>
        <button
          className={activeTab === 'segmentation' ? styles.activeTab : styles.tab}
          onClick={() => setActiveTab('segmentation')}
        >
          Segmentation
        </button>
        <button
          className={activeTab === 'synthetic' ? styles.activeTab : styles.tab}
          onClick={() => setActiveTab('synthetic')}
        >
          Synthetic MRI
        </button>
        <button
          className={activeTab === '3d' ? styles.activeTab : styles.tab}
          onClick={() => setActiveTab('3d')}
        >
          3D Model
        </button>
        <button
          className={activeTab === 'simulation' ? styles.activeTab : styles.tab}
          onClick={() => setActiveTab('simulation')}
        >
          Simulation
        </button>
      </div>

      <div className={styles.tabContent}>
        {activeTab === 'segmentation' && (
          <div className={styles.tabPane}>
            <h3>Segmentation Results</h3>
            {segmentationOutput ? (
              <>
                <div className={styles.metadata}>
                  {segmentationOutput.metadata && (
                    <div>
                      <h4>Metadata</h4>
                      <pre>{JSON.stringify(segmentationOutput.metadata, null, 2)}</pre>
                    </div>
                  )}
                </div>
                {renderArtifacts(segmentationOutput.artifacts)}
                <div className={styles.viewerPlaceholder}>
                  <h4>NIfTI Viewer (Placeholder)</h4>
                  <p>A 3D NIfTI volume viewer would be displayed here.</p>
                  <p>Integration with tools like Niivue or OHIF viewer would go in this section.</p>
                </div>
              </>
            ) : (
              <div className={styles.empty}>
                No segmentation output yet. Run the segmentation step first.
              </div>
            )}
          </div>
        )}

        {activeTab === 'synthetic' && (
          <div className={styles.tabPane}>
            <h3>Synthetic MRI Results</h3>
            {syntheticOutput ? (
              <>
                <div className={styles.metadata}>
                  {syntheticOutput.metadata && (
                    <div>
                      <h4>Metadata</h4>
                      <pre>{JSON.stringify(syntheticOutput.metadata, null, 2)}</pre>
                    </div>
                  )}
                </div>
                {renderArtifacts(syntheticOutput.artifacts)}
                <div className={styles.viewerPlaceholder}>
                  <h4>MRI Viewer (Placeholder)</h4>
                  <p>A multi-modal MRI viewer would display synthetic T1, T1ce, T2, and FLAIR volumes here.</p>
                </div>
              </>
            ) : (
              <div className={styles.empty}>
                No synthetic MRI output yet. Run the diffusion generation step first.
              </div>
            )}
          </div>
        )}

        {activeTab === '3d' && (
          <div className={styles.tabPane}>
            <h3>3D Tumor Model</h3>
            {modelOutput ? (
              <>
                <div className={styles.metadata}>
                  {modelOutput.stats && (
                    <div>
                      <h4>Model Statistics</h4>
                      <ul>
                        <li>Format: {modelOutput.format}</li>
                        {modelOutput.stats.vertices && <li>Vertices: {modelOutput.stats.vertices.toLocaleString()}</li>}
                        {modelOutput.stats.faces && <li>Faces: {modelOutput.stats.faces.toLocaleString()}</li>}
                        {modelOutput.stats.volume && <li>Volume: {modelOutput.stats.volume.toFixed(2)} mm³</li>}
                      </ul>
                    </div>
                  )}
                  {modelOutput.modelUrl && (
                    <div>
                      <h4>Model URL</h4>
                      <a href={modelOutput.modelUrl} target="_blank" rel="noopener noreferrer">
                        {modelOutput.modelUrl}
                      </a>
                    </div>
                  )}
                </div>
                {renderArtifacts(modelOutput.artifacts)}
                <div className={styles.viewerPlaceholder}>
                  <h4>3D Viewer (Placeholder)</h4>
                  <p>A three.js or similar 3D viewer would render the GLB/OBJ model here.</p>
                  <p>Integration example would include rotation, zoom, and material controls.</p>
                  {modelOutput.modelUrl && (
                    <div className={styles.placeholderNote}>
                      Model file: {modelOutput.modelUrl}
                    </div>
                  )}
                </div>
              </>
            ) : (
              <div className={styles.empty}>
                No 3D model output yet. Run the 3D modeling step first.
              </div>
            )}
          </div>
        )}

        {activeTab === 'simulation' && (
          <div className={styles.tabPane}>
            <h3>Therapeutic Agent Simulation</h3>
            {simulationOutput ? (
              <>
                <div className={styles.metadata}>
                  {simulationOutput.metadata && (
                    <div>
                      <h4>Simulation Parameters</h4>
                      <pre>{JSON.stringify(simulationOutput.metadata, null, 2)}</pre>
                    </div>
                  )}
                  {simulationOutput.frames && (
                    <div>
                      <h4>Time Steps ({simulationOutput.frames.length} frames)</h4>
                      <div className={styles.framesList}>
                        {simulationOutput.frames.map((frame: any, idx: number) => (
                          <div key={idx} className={styles.frameItem}>
                            <span>t = {frame.t}s</span>
                            <a href={frame.artifactUrl} target="_blank" rel="noopener noreferrer">
                              View frame
                            </a>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
                {renderArtifacts(simulationOutput.artifacts)}
                <div className={styles.viewerPlaceholder}>
                  <h4>Simulation Visualization (Placeholder)</h4>
                  <p>A heatmap/overlay visualization would show therapeutic agent distribution over time.</p>
                  <p>This could include time-series sliders, colormap controls, and overlay on 3D model.</p>
                </div>
              </>
            ) : (
              <div className={styles.empty}>
                No simulation output yet. Run the simulation step first.
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};
