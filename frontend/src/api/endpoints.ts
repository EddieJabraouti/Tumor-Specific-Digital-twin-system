import {
  JobResponse,
  JobStatusResponse,
  PreprocessRequest,
  SegmentRequest,
  DiffuseRequest,
  Tumor3DRequest,
  SimulateRequest,
} from './types';
import { apiFetch, uploadMultipleFiles, uploadFile } from './apiClient';
import { mockApi } from './mockApi';

export interface ApiConfig {
  baseUrl: string;
  mockMode: boolean;
}

// Read from environment variables (for production builds)
// In .env file: VITE_API_BASE_URL=http://your-backend-url:8000
// In .env.development: VITE_API_BASE_URL=http://localhost:8000
const defaultBaseUrl = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
const defaultMockMode = import.meta.env.DEV; // Mock mode ON in dev by default

let config: ApiConfig = {
  baseUrl: defaultBaseUrl,
  mockMode: defaultMockMode,
};

export function setApiConfig(newConfig: Partial<ApiConfig>) {
  config = { ...config, ...newConfig };
}

export function getApiConfig(): ApiConfig {
  return { ...config };
}

function buildUrl(path: string): string {
  return `${config.baseUrl}${path}`;
}

async function createJob(
  endpoint: string,
  request: any,
  mockFn: (req: any) => Promise<JobResponse>
): Promise<JobResponse> {
  if (config.mockMode) {
    return mockFn(request);
  }

  // In real mode, we'd handle file uploads properly
  // For now, assume the backend accepts multipart/form-data
  const formData = new FormData();
  
  // Handle file uploads
  Object.entries(request).forEach(([key, value]) => {
    if (value instanceof File) {
      formData.append(key, value);
    } else if (value && typeof value === 'object') {
      // Nested objects like modalities
      Object.entries(value).forEach(([subKey, subValue]) => {
        if (subValue instanceof File) {
          formData.append(`${key}.${subKey}`, subValue);
        } else if (typeof subValue === 'string') {
          formData.append(`${key}.${subKey}`, subValue);
        }
      });
    } else if (typeof value === 'string') {
      formData.append(key, value);
    } else if (value !== undefined && value !== null) {
      formData.append(key, JSON.stringify(value));
    }
  });

  return apiFetch<JobResponse>(buildUrl(endpoint), {
    method: 'POST',
    body: formData,
  });
}

export const api = {
  // Preprocess
  async preprocess(request: PreprocessRequest): Promise<JobResponse> {
    return createJob('/api/preprocess', request, mockApi.preprocess);
  },

  // Segment
  async segment(request: SegmentRequest): Promise<JobResponse> {
    return createJob('/api/segment', request, mockApi.segment);
  },

  // Diffuse
  async diffuse(request: DiffuseRequest): Promise<JobResponse> {
    return createJob('/api/diffuse', request, mockApi.diffuse);
  },

  // 3D Model
  async tumor3d(request: Tumor3DRequest): Promise<JobResponse> {
    return createJob('/api/tumor3d', request, mockApi.tumor3d);
  },

  // Simulate
  async simulate(request: SimulateRequest): Promise<JobResponse> {
    return createJob('/api/simulate', request, mockApi.simulate);
  },

  // Get job status
  async getJobStatus(jobId: string): Promise<JobStatusResponse> {
    if (config.mockMode) {
      return mockApi.getJobStatus(jobId);
    }

    return apiFetch<JobStatusResponse>(buildUrl(`/api/jobs/${jobId}`));
  },
};
