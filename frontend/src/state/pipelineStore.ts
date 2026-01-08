import { create } from 'zustand';
import { JobStatus } from '../api/types';

export type StepId =
  | 'preprocess'
  | 'segment'
  | 'diffuse'
  | 're-segment'
  | 'tumor3d'
  | 'simulate';

export interface StepState {
  status: 'idle' | 'running' | 'success' | 'error';
  jobId?: string;
  progress: number;
  error?: string;
  result?: any;
}

export interface LogEntry {
  timestamp: string;
  step: StepId | 'system';
  level: 'info' | 'warn' | 'error' | 'success';
  message: string;
}

export interface UploadedFiles {
  t1?: File;
  t1ce?: File;
  t2?: File;
  flair?: File;
  mask?: File;
}

interface PipelineState {
  // File inputs
  uploadedFiles: UploadedFiles;

  // Step states
  steps: Record<StepId, StepState>;

  // Output artifacts (keyed by step)
  outputs: Record<StepId, any>;

  // Run logs
  logs: LogEntry[];

  // Pipeline control
  isRunningPipeline: boolean;
  pipelineCancelled: boolean;

  // Actions
  setUploadedFile: (modality: keyof UploadedFiles, file: File | undefined) => void;
  setStepStatus: (step: StepId, status: StepState['status'], data?: Partial<StepState>) => void;
  setStepProgress: (step: StepId, progress: number) => void;
  setStepResult: (step: StepId, result: any) => void;
  addLog: (step: StepId | 'system', level: LogEntry['level'], message: string) => void;
  clearLogs: () => void;
  setPipelineRunning: (running: boolean) => void;
  setPipelineCancelled: (cancelled: boolean) => void;
  reset: () => void;
}

const initialStepState: StepState = {
  status: 'idle',
  progress: 0,
};

const initialState: Omit<PipelineState, keyof {
  setUploadedFile: any;
  setStepStatus: any;
  setStepProgress: any;
  setStepResult: any;
  addLog: any;
  clearLogs: any;
  setPipelineRunning: any;
  setPipelineCancelled: any;
  reset: any;
}> = {
  uploadedFiles: {},
  steps: {
    preprocess: { ...initialStepState },
    segment: { ...initialStepState },
    diffuse: { ...initialStepState },
    're-segment': { ...initialStepState },
    tumor3d: { ...initialStepState },
    simulate: { ...initialStepState },
  },
  outputs: {},
  logs: [],
  isRunningPipeline: false,
  pipelineCancelled: false,
};

export const usePipelineStore = create<PipelineState>((set) => ({
  ...initialState,

  setUploadedFile: (modality, file) =>
    set((state) => ({
      uploadedFiles: {
        ...state.uploadedFiles,
        [modality]: file,
      },
    })),

  setStepStatus: (step, status, data = {}) =>
    set((state) => ({
      steps: {
        ...state.steps,
        [step]: {
          ...state.steps[step],
          status,
          ...data,
        },
      },
    })),

  setStepProgress: (step, progress) =>
    set((state) => ({
      steps: {
        ...state.steps,
        [step]: {
          ...state.steps[step],
          progress,
        },
      },
    })),

  setStepResult: (step, result) =>
    set((state) => ({
      steps: {
        ...state.steps,
        [step]: {
          ...state.steps[step],
          status: 'success',
          result,
        },
      },
      outputs: {
        ...state.outputs,
        [step]: result,
      },
    })),

  addLog: (step, level, message) =>
    set((state) => ({
      logs: [
        ...state.logs,
        {
          timestamp: new Date().toISOString(),
          step,
          level,
          message,
        },
      ],
    })),

  clearLogs: () => set({ logs: [] }),

  setPipelineRunning: (isRunningPipeline) =>
    set({ isRunningPipeline, pipelineCancelled: false }),

  setPipelineCancelled: (pipelineCancelled) =>
    set({ pipelineCancelled, isRunningPipeline: false }),

  reset: () =>
    set({
      ...initialState,
      uploadedFiles: {},
    }),
}));
