import React, { useEffect, useRef } from 'react';
import { usePipelineStore, StepId } from '../state/pipelineStore';
import { StatusBadge } from './StatusBadge';
import { RunLog } from './RunLog';
import { api, setApiConfig, getApiConfig } from '../api/endpoints';
import styles from './PipelinePanel.module.css';

const STEP_CONFIG: Array<{
  id: StepId;
  name: string;
  description: string;
  requiresInputs: boolean;
}> = [
  { id: 'preprocess', name: 'Preprocess', description: 'Preprocess MRI modalities', requiresInputs: true },
  { id: 'segment', name: 'Segment Tumor', description: 'U-Net segmentation', requiresInputs: true },
  { id: 'diffuse', name: 'Generate Synthetic MRI', description: 'Conditional diffusion generation', requiresInputs: false },
  { id: 're-segment', name: 'Re-segment Synthetic MRI', description: 'Validate with segmentation', requiresInputs: false },
  { id: 'tumor3d', name: 'Create 3D Tumor Model', description: '3D reconstruction', requiresInputs: false },
  { id: 'simulate', name: 'Simulate Therapeutic Agent', description: 'Physics-powered NN simulation', requiresInputs: false },
];

// Polling function for job status
async function pollJobStatus(
  jobId: string,
  onProgress: (progress: number) => void,
  onSuccess: (result: any) => void,
  onError: (error: string) => void,
  onCancel: () => boolean
): Promise<void> {
  const pollInterval = 1000; // 1 second
  const maxAttempts = 300; // 5 minutes max
  let attempts = 0;

  const poll = async () => {
    if (onCancel()) {
      return;
    }

    try {
      const status = await api.getJobStatus(jobId);
      onProgress(status.progress);

      if (status.status === 'succeeded') {
        onSuccess(status.result);
        return;
      }

      if (status.status === 'failed') {
        onError(status.error || 'Job failed');
        return;
      }

      attempts++;
      if (attempts >= maxAttempts) {
        onError('Job timeout');
        return;
      }

      setTimeout(poll, pollInterval);
    } catch (error) {
      onError(error instanceof Error ? error.message : 'Unknown error');
    }
  };

  poll();
}

export const PipelinePanel: React.FC = () => {
  const {
    uploadedFiles,
    steps,
    isRunningPipeline,
    pipelineCancelled,
    setStepStatus,
    setStepProgress,
    setStepResult,
    addLog,
    setPipelineRunning,
    setPipelineCancelled,
    reset,
  } = usePipelineStore();

  const cancelRef = useRef(false);

  const canRunStep = (step: StepId, requiresInputs: boolean): boolean => {
    if (requiresInputs) {
      const hasAllModalities =
        uploadedFiles.t1 && uploadedFiles.t1ce && uploadedFiles.t2 && uploadedFiles.flair;
      return !!hasAllModalities;
    }

    // For steps that don't require inputs, check if previous steps succeeded
    switch (step) {
      case 'diffuse':
        return steps.segment.status === 'success';
      case 're-segment':
        return steps.diffuse.status === 'success';
      case 'tumor3d':
        return steps.segment.status === 'success' || steps['re-segment'].status === 'success';
      case 'simulate':
        return steps.tumor3d.status === 'success';
      default:
        return true;
    }
  };

  const runStep = async (step: StepId) => {
    cancelRef.current = false;
    setStepStatus(step, 'running', { progress: 0, error: undefined });
    addLog(step, 'info', `Starting ${STEP_CONFIG.find((s) => s.id === step)?.name}...`);

    try {
      let jobId: string;

      switch (step) {
        case 'preprocess':
          jobId = (await api.preprocess({
            modalities: {
              t1: uploadedFiles.t1!,
              t1ce: uploadedFiles.t1ce!,
              t2: uploadedFiles.t2!,
              flair: uploadedFiles.flair!,
            },
            mask: uploadedFiles.mask,
          })).jobId;
          break;

        case 'segment':
          // Use outputs from preprocess if available, otherwise use uploaded files
          const segmentModalities = steps.preprocess.status === 'success'
            ? {
                t1: steps.preprocess.result?.artifacts?.[0]?.url || uploadedFiles.t1!,
                t1ce: steps.preprocess.result?.artifacts?.[1]?.url || uploadedFiles.t1ce!,
                t2: steps.preprocess.result?.artifacts?.[2]?.url || uploadedFiles.t2!,
                flair: steps.preprocess.result?.artifacts?.[3]?.url || uploadedFiles.flair!,
              }
            : {
                t1: uploadedFiles.t1!,
                t1ce: uploadedFiles.t1ce!,
                t2: uploadedFiles.t2!,
                flair: uploadedFiles.flair!,
              };

          jobId = (await api.segment({
            modalities: segmentModalities,
            mask: uploadedFiles.mask,
          })).jobId;
          break;

        case 'diffuse':
          const diffuseModalities = steps.preprocess.status === 'success'
            ? {
                t1: steps.preprocess.result?.artifacts?.[0]?.url,
                t1ce: steps.preprocess.result?.artifacts?.[1]?.url,
                t2: steps.preprocess.result?.artifacts?.[2]?.url,
                flair: steps.preprocess.result?.artifacts?.[3]?.url,
              }
            : {
                t1: uploadedFiles.t1!,
                t1ce: uploadedFiles.t1ce!,
                t2: uploadedFiles.t2!,
                flair: uploadedFiles.flair!,
              };

          jobId = (await api.diffuse({
            modalities: diffuseModalities,
            segmentationMask: steps.segment.result?.artifacts?.[0]?.url || uploadedFiles.mask!,
          })).jobId;
          break;

        case 're-segment':
          jobId = (await api.segment({
            modalities: {
              t1: steps.diffuse.result?.artifacts?.[0]?.url,
              t1ce: steps.diffuse.result?.artifacts?.[1]?.url,
              t2: steps.diffuse.result?.artifacts?.[2]?.url,
              flair: steps.diffuse.result?.artifacts?.[3]?.url,
            },
          })).jobId;
          break;

        case 'tumor3d':
          const maskFor3d =
            steps['re-segment'].status === 'success'
              ? steps['re-segment'].result?.artifacts?.[0]?.url
              : steps.segment.result?.artifacts?.[0]?.url ||
                (uploadedFiles.mask instanceof File ? 'file' : uploadedFiles.mask);

          jobId = (await api.tumor3d({
            segmentationMask: maskFor3d,
          })).jobId;
          break;

        case 'simulate':
          jobId = (await api.simulate({
            tumorModel: steps.tumor3d.result?.modelUrl || steps.tumor3d.result?.artifacts?.[0]?.url,
            segmentationMask:
              steps['re-segment'].status === 'success'
                ? steps['re-segment'].result?.artifacts?.[0]?.url
                : steps.segment.result?.artifacts?.[0]?.url,
          })).jobId;
          break;

        default:
          throw new Error(`Unknown step: ${step}`);
      }

      setStepStatus(step, 'running', { jobId });

      await pollJobStatus(
        jobId,
        (progress) => {
          setStepProgress(step, progress);
        },
        (result) => {
          setStepResult(step, result);
          addLog(step, 'success', `${STEP_CONFIG.find((s) => s.id === step)?.name} completed successfully`);
        },
        (error) => {
          setStepStatus(step, 'error', { error });
          addLog(step, 'error', `${STEP_CONFIG.find((s) => s.id === step)?.name} failed: ${error}`);
        },
        () => cancelRef.current || pipelineCancelled
      );
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      setStepStatus(step, 'error', { error: errorMessage });
      addLog(step, 'error', `Failed to start ${STEP_CONFIG.find((s) => s.id === step)?.name}: ${errorMessage}`);
    }
  };

  const runFullPipeline = async () => {
    setPipelineRunning(true);
    cancelRef.current = false;
    addLog('system', 'info', 'Starting full pipeline...');

    for (const stepConfig of STEP_CONFIG) {
      if (cancelRef.current || pipelineCancelled) {
        addLog('system', 'warn', 'Pipeline cancelled by user');
        break;
      }

      if (!canRunStep(stepConfig.id, stepConfig.requiresInputs)) {
        addLog('system', 'warn', `Skipping ${stepConfig.name}: prerequisites not met`);
        continue;
      }

      await runStep(stepConfig.id);

      if (steps[stepConfig.id].status === 'error') {
        addLog('system', 'error', `Pipeline stopped at ${stepConfig.name} due to error`);
        break;
      }
    }

    setPipelineRunning(false);
    if (!cancelRef.current && !pipelineCancelled) {
      addLog('system', 'success', 'Full pipeline completed');
    }
  };

  const cancelPipeline = () => {
    cancelRef.current = true;
    setPipelineCancelled(true);
    addLog('system', 'warn', 'Pipeline cancellation requested');
  };

  const handleReset = () => {
    reset();
    cancelRef.current = false;
  };

  return (
    <div className={styles.panel}>
      <h2>Pipeline Steps</h2>

      <div className={styles.controls}>
        <button
          onClick={runFullPipeline}
          disabled={isRunningPipeline || !canRunStep('preprocess', true)}
          className={styles.fullPipelineButton}
        >
          {isRunningPipeline ? 'Running...' : 'Run Full Pipeline'}
        </button>
        {isRunningPipeline && (
          <button onClick={cancelPipeline} className={styles.cancelButton}>
            Cancel
          </button>
        )}
        <button onClick={handleReset} className={styles.resetButton}>
          Reset
        </button>
      </div>

      <div className={styles.steps}>
        {STEP_CONFIG.map((stepConfig) => {
          const step = steps[stepConfig.id];
          const canRun = canRunStep(stepConfig.id, stepConfig.requiresInputs);

          return (
            <div key={stepConfig.id} className={styles.step}>
              <div className={styles.stepHeader}>
                <div className={styles.stepInfo}>
                  <h3>{stepConfig.name}</h3>
                  <p className={styles.description}>{stepConfig.description}</p>
                </div>
                <StatusBadge status={step.status} progress={step.progress} />
              </div>
              <button
                onClick={() => runStep(stepConfig.id)}
                disabled={!canRun || step.status === 'running' || isRunningPipeline}
                className={styles.runButton}
              >
                Run Step
              </button>
              {step.error && (
                <div className={styles.error}>{step.error}</div>
              )}
            </div>
          );
        })}
      </div>

      <RunLog />
    </div>
  );
};
