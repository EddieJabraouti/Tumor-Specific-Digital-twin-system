# Glioblastoma Digital Twin - Frontend

A React + TypeScript frontend application for the Glioblastoma Digital Twin research pipeline. This frontend provides a clean UI to run a modular pipeline for MRI preprocessing, tumor segmentation, synthetic MRI generation, 3D reconstruction, and therapeutic agent simulation.

## Quick Start

### Prerequisites

- Node.js 18+ and npm/yarn/pnpm

### Installation

```bash
cd frontend
npm install
```

### Development

```bash
npm run dev
```

The app will be available at `http://localhost:5173` (Vite default port).

### Build

```bash
npm run build
```

Output will be in the `dist/` directory.

### Preview Production Build

```bash
npm run preview
```

## Mock Mode vs Real Mode

The frontend supports two modes of operation:

### Mock Mode (Default)

When **Mock Mode** is enabled (checked in the Backend Configuration panel):
- All API calls use local mock functions in `src/api/mockApi.ts`
- Responses are simulated with artificial delays (2-8 seconds per step)
- No actual backend is required
- Perfect for development and testing the UI without backend infrastructure

### Real Mode

When **Mock Mode** is disabled:
- API calls are made to the backend URL specified in the Base URL field (default: `http://localhost:8000`)
- All requests use real HTTP calls via `fetch`
- Backend must be running and implement the expected API endpoints

Toggle between modes using the checkbox in the **Backend Configuration** panel at the bottom of the page.

## Expected Backend API Endpoints

The backend should implement the following REST endpoints:

### Job-Based API Pattern

All model operations follow an async job pattern:
1. **POST** to start a job → returns `{ jobId: string }`
2. **GET** `/api/jobs/:jobId` → returns job status and progress
3. Poll until status is `succeeded` or `failed`

### Endpoints

#### 1. Preprocess
- **POST** `/api/preprocess`
- **Payload**: Multipart form data with files:
  - `modalities.t1`: File (NIfTI)
  - `modalities.t1ce`: File (NIfTI)
  - `modalities.t2`: File (NIfTI)
  - `modalities.flair`: File (NIfTI)
  - `mask`: File (optional, NIfTI)
- **Response**: `{ jobId: string }`
- **Result**: `{ artifacts: Artifact[], metadata?: object }`

#### 2. Segmentation
- **POST** `/api/segment`
- **Payload**: Multipart form data:
  - `modalities.t1`, `modalities.t1ce`, `modalities.t2`, `modalities.flair`: File or string (artifact URL from previous step)
  - `mask`: File or string (optional)
- **Response**: `{ jobId: string }`
- **Result**: `{ artifacts: Artifact[], metadata?: { tumorVolume?: number, classDistribution?: object } }`

#### 3. Diffusion (Synthetic MRI Generation)
- **POST** `/api/diffuse`
- **Payload**: Multipart form data:
  - `modalities.t1`, `modalities.t1ce`, `modalities.t2`, `modalities.flair`: File or string
  - `segmentationMask`: File or string (required for conditional generation)
- **Response**: `{ jobId: string }`
- **Result**: `{ artifacts: Artifact[], metadata?: object }`

#### 4. 3D Tumor Modeling
- **POST** `/api/tumor3d`
- **Payload**: Multipart form data:
  - `segmentationMask`: File or string
- **Response**: `{ jobId: string }`
- **Result**: `{ artifacts: Artifact[], modelUrl: string, format: "glb" | "obj", stats?: { vertices?: number, faces?: number, volume?: number } }`

#### 5. Physics Simulation
- **POST** `/api/simulate`
- **Payload**: Multipart form data:
  - `tumorModel`: File or string (3D model from previous step)
  - `segmentationMask`: File or string (optional)
  - `parameters`: JSON string (optional, object with simulation parameters)
- **Response**: `{ jobId: string }`
- **Result**: `{ artifacts: Artifact[], frames: Array<{ t: number, artifactUrl: string }>, metadata?: object }`

#### 6. Job Status
- **GET** `/api/jobs/:jobId`
- **Response**:
```typescript
{
  status: "queued" | "running" | "succeeded" | "failed",
  progress: number,  // 0-100
  result?: {
    artifacts: Array<{
      name: string,
      url: string,
      kind: "nifti" | "json" | "mesh" | "image" | "other"
    }>,
    metadata?: object,
    // Step-specific fields (see above)
  },
  error?: string
}
```

## Job Schema Details

### Artifact Structure

Each successful job returns artifacts:

```typescript
interface Artifact {
  name: string;        // Display name (e.g., "segmentation_mask.nii.gz")
  url: string;         // URL to download/view the artifact
  kind: "nifti" | "json" | "mesh" | "image" | "other";
}
```

### Status Values

- `queued`: Job accepted, waiting to start
- `running`: Job is executing (progress should increment)
- `succeeded`: Job completed successfully (result contains outputs)
- `failed`: Job failed (error contains failure reason)

## Output Chaining Between Steps

The pipeline chains outputs from previous steps to subsequent steps:

1. **Preprocess → Segment**: Uses preprocessed artifact URLs if available, otherwise uses uploaded files
2. **Segment → Diffuse**: Uses segmentation mask artifact from segment step
3. **Diffuse → Re-segment**: Uses synthetic MRI artifacts from diffuse step
4. **Segment/Re-segment → Tumor3D**: Uses segmentation mask from most recent segmentation
5. **Tumor3D → Simulate**: Uses 3D model URL from tumor3d step

The frontend automatically passes artifact URLs (from previous step results) or File objects (from uploads) depending on what's available.

## Project Structure

```
frontend/
├── src/
│   ├── api/
│   │   ├── apiClient.ts      # Fetch wrapper with error handling
│   │   ├── endpoints.ts       # API endpoint functions
│   │   ├── mockApi.ts         # Mock implementations with delays
│   │   └── types.ts           # TypeScript interfaces
│   ├── components/
│   │   ├── BackendConfigPanel.tsx
│   │   ├── InputsPanel.tsx
│   │   ├── OutputsPanel.tsx
│   │   ├── PipelinePanel.tsx
│   │   ├── RunLog.tsx
│   │   ├── StatusBadge.tsx
│   │   └── *.module.css       # Component styles
│   ├── state/
│   │   └── pipelineStore.ts   # Zustand state management
│   ├── App.tsx                # Main app component
│   ├── main.tsx               # Entry point
│   └── index.css              # Global styles
├── index.html
├── package.json
├── tsconfig.json
├── vite.config.ts
└── README.md
```

## Key Features

- **Single-page application**: All functionality on one page with clear sections
- **Job polling**: Automatically polls job status until completion
- **Pipeline orchestration**: Run individual steps or the full pipeline sequentially
- **Error handling**: Comprehensive error states and user feedback
- **Logging**: Timestamped run logs for all operations
- **File upload**: Support for NIfTI file uploads (T1, T1ce, T2, FLAIR, masks)
- **Output visualization placeholders**: Structured UI for viewing results (actual viewers to be integrated later)

## Switching to Real Backend

To connect to a real backend:

1. Ensure your backend implements all endpoints listed above
2. Start your backend server (e.g., on `http://localhost:8000`)
3. Open the frontend app
4. Scroll to the **Backend Configuration** panel
5. Uncheck **Mock Mode**
6. Update **Base URL** if your backend runs on a different port/host
7. The frontend will now make real API calls

## Notes

- The frontend assumes async jobs that may take time to complete
- Job polling interval: 1 second
- Maximum polling duration: 5 minutes per job
- File uploads use multipart/form-data
- All API responses should be JSON (except file artifacts)
- The UI disables step buttons when prerequisites are not met

## Development Notes

- Uses **Zustand** for state management (lightweight, simple)
- **Vite** for fast development and building
- **TypeScript** for type safety
- **CSS Modules** for scoped styling
- No heavy UI frameworks - plain React with minimal styling

## Future Enhancements

- Integrate actual NIfTI viewers (e.g., Niivue, OHIF)
- Add three.js 3D model viewer for GLB/OBJ files
- Implement simulation visualization with time-series controls
- Add export/download functionality for results
- Support for batch processing multiple patients
