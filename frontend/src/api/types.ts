// Job status types
export type JobStatus = 'queued' | 'running' | 'succeeded' | 'failed';

// Artifact types
export type ArtifactKind = 'nifti' | 'json' | 'mesh' | 'image' | 'other';

export interface Artifact {
  name: string;
  url: string;
  kind: ArtifactKind;
}

// Job response types
export interface JobResponse {
  jobId: string;
}

export interface JobStatusResponse {
  status: JobStatus;
  progress: number;
  result?: any;
  error?: string;
}

// Preprocess API
export interface PreprocessRequest {
  modalities: {
    t1?: File;
    t1ce?: File;
    t2?: File;
    flair?: File;
  };
  mask?: File;
}

export interface PreprocessResult {
  artifacts: Artifact[];
  metadata?: Record<string, any>;
}

// Segmentation API
export interface SegmentRequest {
  modalities: {
    t1?: File | string; // File or artifact URL from previous step
    t1ce?: File | string;
    t2?: File | string;
    flair?: File | string;
  };
  mask?: File | string;
}

export interface SegmentResult {
  artifacts: Artifact[];
  metadata?: {
    tumorVolume?: number;
    classDistribution?: Record<string, number>;
  };
}

// Diffusion API
export interface DiffuseRequest {
  modalities: {
    t1?: File | string;
    t1ce?: File | string;
    t2?: File | string;
    flair?: File | string;
  };
  segmentationMask: File | string; // Required for conditional generation
}

export interface DiffuseResult {
  artifacts: Artifact[];
  metadata?: Record<string, any>;
}

// 3D Modeling API
export interface Tumor3DRequest {
  segmentationMask: File | string;
}

export interface Tumor3DResult {
  artifacts: Artifact[];
  modelUrl: string;
  format: 'glb' | 'obj';
  stats?: {
    vertices?: number;
    faces?: number;
    volume?: number;
  };
}

// Physics Simulation API
export interface SimulateRequest {
  tumorModel: File | string; // 3D model from previous step
  segmentationMask?: File | string;
  parameters?: Record<string, any>;
}

export interface SimulateResult {
  artifacts: Artifact[];
  frames: Array<{
    t: number;
    artifactUrl: string;
  }>;
  metadata?: Record<string, any>;
}
