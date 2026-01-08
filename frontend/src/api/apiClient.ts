import { JobResponse, JobStatusResponse } from './types';

export class ApiError extends Error {
  constructor(
    message: string,
    public status?: number,
    public response?: any
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

const DEFAULT_TIMEOUT = 30000; // 30 seconds

interface RequestOptions {
  timeout?: number;
  signal?: AbortSignal;
}

export async function apiFetch<T>(
  url: string,
  options: RequestInit & RequestOptions = {}
): Promise<T> {
  const { timeout = DEFAULT_TIMEOUT, signal, ...fetchOptions } = options;

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeout);

  // Combine abort signals if both provided
  if (signal) {
    signal.addEventListener('abort', () => controller.abort());
  }

  try {
    const response = await fetch(url, {
      ...fetchOptions,
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      let errorData;
      try {
        errorData = await response.json();
      } catch {
        errorData = { message: response.statusText };
      }
      throw new ApiError(
        errorData.message || `HTTP ${response.status}`,
        response.status,
        errorData
      );
    }

    // Handle empty responses
    const contentType = response.headers.get('content-type');
    if (!contentType || !contentType.includes('application/json')) {
      return {} as T;
    }

    return await response.json();
  } catch (error) {
    clearTimeout(timeoutId);
    if (error instanceof ApiError) {
      throw error;
    }
    if (error instanceof Error && error.name === 'AbortError') {
      throw new ApiError('Request timeout', 408);
    }
    throw new ApiError(
      error instanceof Error ? error.message : 'Unknown error occurred'
    );
  }
}

export async function uploadFile(
  url: string,
  file: File,
  fieldName: string = 'file',
  additionalData?: Record<string, any>
): Promise<any> {
  const formData = new FormData();
  formData.append(fieldName, file);

  if (additionalData) {
    Object.entries(additionalData).forEach(([key, value]) => {
      if (value !== undefined && value !== null) {
        if (value instanceof File) {
          formData.append(key, value);
        } else {
          formData.append(key, JSON.stringify(value));
        }
      }
    });
  }

  return apiFetch(url, {
    method: 'POST',
    body: formData,
    timeout: 60000, // Longer timeout for file uploads
  });
}

export async function uploadMultipleFiles(
  url: string,
  files: Record<string, File | string>,
  additionalData?: Record<string, any>
): Promise<any> {
  const formData = new FormData();

  Object.entries(files).forEach(([key, value]) => {
    if (value instanceof File) {
      formData.append(key, value);
    } else if (typeof value === 'string') {
      // If it's already an artifact URL, send as JSON
      formData.append(key, value);
    }
  });

  if (additionalData) {
    Object.entries(additionalData).forEach(([key, value]) => {
      if (value !== undefined && value !== null) {
        formData.append(key, JSON.stringify(value));
      }
    });
  }

  return apiFetch(url, {
    method: 'POST',
    body: formData,
    timeout: 60000,
  });
}
