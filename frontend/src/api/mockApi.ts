import {
  JobResponse,
  JobStatusResponse,
  PreprocessRequest,
  SegmentRequest,
  DiffuseRequest,
  Tumor3DRequest,
  SimulateRequest,
  PreprocessResult,
  SegmentResult,
  DiffuseResult,
  Tumor3DResult,
  SimulateResult,
} from './types';

// Mock job storage (in-memory)
const mockJobs = new Map<string, { status: string; progress: number; result?: any; error?: string }>();

// Generate deterministic job IDs
let jobCounter = 0;
function generateJobId(): string {
  return `mock-job-${++jobCounter}`;
}

// Simulate async work with delay
function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// Simulate job progression
async function simulateJob(
  jobId: string,
  resultFactory: () => any,
  duration: number = 3000
): Promise<any> {
  mockJobs.set(jobId, { status: 'queued', progress: 0 });
  await delay(500);

  mockJobs.set(jobId, { status: 'running', progress: 10 });
  await delay(duration * 0.2);

  mockJobs.set(jobId, { status: 'running', progress: 40 });
  await delay(duration * 0.3);

  mockJobs.set(jobId, { status: 'running', progress: 70 });
  await delay(duration * 0.3);

  mockJobs.set(jobId, { status: 'running', progress: 90 });
  await delay(duration * 0.2);

  const result = resultFactory();
  mockJobs.set(jobId, { status: 'succeeded', progress: 100, result });
  return result;
}

export const mockApi = {
  // Preprocess
  async preprocess(_request: PreprocessRequest): Promise<JobResponse> {
    const jobId = generateJobId();
    simulateJob(
      jobId,
      () => ({
        artifacts: [
          { name: 'preprocessed_t1.nii.gz', url: '/mock/artifacts/preprocessed_t1.nii.gz', kind: 'nifti' as const },
          { name: 'preprocessed_t1ce.nii.gz', url: '/mock/artifacts/preprocessed_t1ce.nii.gz', kind: 'nifti' as const },
          { name: 'preprocessed_t2.nii.gz', url: '/mock/artifacts/preprocessed_t2.nii.gz', kind: 'nifti' as const },
          { name: 'preprocessed_flair.nii.gz', url: '/mock/artifacts/preprocessed_flair.nii.gz', kind: 'nifti' as const },
          { name: 'preprocess_metadata.json', url: '/mock/artifacts/preprocess_metadata.json', kind: 'json' as const },
        ],
        metadata: {
          voxelSize: [1.0, 1.0, 1.0],
          dimensions: [240, 240, 155],
          normalized: true,
        },
      } as PreprocessResult),
      2000
    );
    return { jobId };
  },

  // Segment
  async segment(_request: SegmentRequest): Promise<JobResponse> {
    const jobId = generateJobId();
    simulateJob(
      jobId,
      () => ({
        artifacts: [
          { name: 'segmentation_mask.nii.gz', url: '/mock/artifacts/segmentation_mask.nii.gz', kind: 'nifti' as const },
          { name: 'segmentation_metadata.json', url: '/mock/artifacts/segmentation_metadata.json', kind: 'json' as const },
        ],
        metadata: {
          tumorVolume: 45678.5,
          classDistribution: {
            background: 0.85,
            edema: 0.10,
            non_enhancing: 0.03,
            enhancing: 0.02,
          },
        },
      } as SegmentResult),
      4000
    );
    return { jobId };
  },

  // Diffuse (generate synthetic MRI)
  async diffuse(_request: DiffuseRequest): Promise<JobResponse> {
    const jobId = generateJobId();
    simulateJob(
      jobId,
      () => ({
        artifacts: [
          { name: 'synthetic_t1.nii.gz', url: '/mock/artifacts/synthetic_t1.nii.gz', kind: 'nifti' as const },
          { name: 'synthetic_t1ce.nii.gz', url: '/mock/artifacts/synthetic_t1ce.nii.gz', kind: 'nifti' as const },
          { name: 'synthetic_t2.nii.gz', url: '/mock/artifacts/synthetic_t2.nii.gz', kind: 'nifti' as const },
          { name: 'synthetic_flair.nii.gz', url: '/mock/artifacts/synthetic_flair.nii.gz', kind: 'nifti' as const },
          { name: 'diffusion_metadata.json', url: '/mock/artifacts/diffusion_metadata.json', kind: 'json' as const },
        ],
        metadata: {
          steps: 1000,
          guidanceScale: 7.5,
        },
      } as DiffuseResult),
      8000
    );
    return { jobId };
  },

  // 3D Model
  async tumor3d(_request: Tumor3DRequest): Promise<JobResponse> {
    const jobId = generateJobId();
    simulateJob(
      jobId,
      () => ({
        artifacts: [
          { name: 'tumor_model.glb', url: '/mock/artifacts/tumor_model.glb', kind: 'mesh' as const },
          { name: 'tumor_metadata.json', url: '/mock/artifacts/tumor_metadata.json', kind: 'json' as const },
        ],
        modelUrl: '/mock/artifacts/tumor_model.glb',
        format: 'glb' as const,
        stats: {
          vertices: 12543,
          faces: 24321,
          volume: 45678.5,
        },
      } as Tumor3DResult),
      3000
    );
    return { jobId };
  },

  // Simulate (physics simulation)
  async simulate(_request: SimulateRequest): Promise<JobResponse> {
    const jobId = generateJobId();
    simulateJob(
      jobId,
      () => ({
        artifacts: [
          { name: 'simulation_results.json', url: '/mock/artifacts/simulation_results.json', kind: 'json' as const },
        ],
        frames: Array.from({ length: 20 }, (_, i) => ({
          t: i * 0.5,
          artifactUrl: `/mock/artifacts/simulation_frame_${i}.nii.gz`,
        })),
        metadata: {
          timesteps: 20,
          dt: 0.5,
          parameters: {
            diffusionCoefficient: 0.001,
            decayRate: 0.05,
          },
        },
      } as SimulateResult),
      5000
    );
    return { jobId };
  },

  // Get job status
  async getJobStatus(jobId: string): Promise<JobStatusResponse> {
    await delay(100); // Small delay to simulate network
    const job = mockJobs.get(jobId);
    if (!job) {
      throw new Error(`Job ${jobId} not found`);
    }
    return job as JobStatusResponse;
  },
};
