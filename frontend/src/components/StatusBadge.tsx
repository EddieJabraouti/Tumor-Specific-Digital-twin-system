import React from 'react';
import styles from './StatusBadge.module.css';

interface StatusBadgeProps {
  status: 'idle' | 'running' | 'success' | 'error';
  progress?: number;
}

export const StatusBadge: React.FC<StatusBadgeProps> = ({ status, progress }) => {
  const getLabel = () => {
    if (status === 'running' && progress !== undefined) {
      return `${status} (${progress}%)`;
    }
    return status;
  };

  return (
    <span className={`${styles.badge} ${styles[status]}`}>
      {getLabel()}
    </span>
  );
};
