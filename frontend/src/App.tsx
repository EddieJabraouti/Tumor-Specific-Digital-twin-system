import React from 'react';
import { InputsPanel } from './components/InputsPanel';
import { PipelinePanel } from './components/PipelinePanel';
import { OutputsPanel } from './components/OutputsPanel';
import { BackendConfigPanel } from './components/BackendConfigPanel';
import styles from './App.module.css';

function App() {
  return (
    <div className={styles.app}>
      <header className={styles.header}>
        <h1>Glioblastoma Digital Twin</h1>
        <p className={styles.subtitle}>Research pipeline UI (not for clinical use)</p>
        <p className={styles.disclaimer}>
          This tool is for research purposes only. Not intended for clinical diagnosis or treatment decisions.
        </p>
      </header>

      <main className={styles.main}>
        <div className={styles.container}>
          <section className={styles.section}>
            <InputsPanel />
          </section>

          <section className={styles.section}>
            <PipelinePanel />
          </section>

          <section className={styles.section}>
            <OutputsPanel />
          </section>

          {import.meta.env.DEV && (
            <section className={styles.section}>
              <BackendConfigPanel />
            </section>
          )}
        </div>
      </main>
    </div>
  );
}

export default App;
