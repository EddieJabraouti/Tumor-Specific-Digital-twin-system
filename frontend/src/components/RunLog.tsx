import React from 'react';
import { usePipelineStore } from '../state/pipelineStore';
import styles from './RunLog.module.css';

export const RunLog: React.FC = () => {
  const logs = usePipelineStore((state) => state.logs);

  const formatTimestamp = (iso: string) => {
    const date = new Date(iso);
    return date.toLocaleTimeString();
  };

  if (logs.length === 0) {
    return (
      <div className={styles.container}>
        <div className={styles.empty}>No logs yet. Run a step to see logs.</div>
      </div>
    );
  }

  return (
    <div className={styles.container}>
      <div className={styles.header}>Run Log</div>
      <div className={styles.logs}>
        {logs.map((log, idx) => (
          <div key={idx} className={`${styles.logEntry} ${styles[log.level]}`}>
            <span className={styles.timestamp}>{formatTimestamp(log.timestamp)}</span>
            <span className={styles.step}>[{log.step}]</span>
            <span className={styles.message}>{log.message}</span>
          </div>
        ))}
      </div>
    </div>
  );
};
