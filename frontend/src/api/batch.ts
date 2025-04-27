import { AudioFeatureType, AudioAnalysisResponse } from '../types';
import { api } from './index';

/**
 * Attempts to call a function with retry logic on failure
 */
async function withRetry<T>(
  fn: () => Promise<T>,
  retries: number = 3,
  delay: number = 1000
): Promise<T> {
  try {
    return await fn();
  } catch (error) {
    if (retries <= 0) {
      throw error;
    }
    
    // Wait before retrying
    await new Promise(resolve => setTimeout(resolve, delay));
    
    // Retry with exponential backoff
    console.log(`Retrying operation. Attempts remaining: ${retries}`);
    return withRetry(fn, retries - 1, delay * 2);
  }
}

/**
 * Analyze multiple audio files in batch
 */
export const analyzeAudioBatch = async (
  files: File[],
  featureTypes: AudioFeatureType[],
  waitForCompletion: boolean = false
): Promise<AudioAnalysisResponse[]> => {
  const formData = new FormData();
  
  // Add all files to the form data
  files.forEach(file => {
    formData.append('files', file);
  });
  
  // Convert feature types to strings
  const featureTypesStr = JSON.stringify(featureTypes.map(ft => ft.toString()));
  console.log('Sending feature types for batch:', featureTypesStr);
  formData.append('feature_types', featureTypesStr);
  formData.append('chunk_duration', '60.0');
  formData.append('wait_for_completion', waitForCompletion.toString());

  try {
    // Use retry mechanism for the API call
    const response = await withRetry(() => 
      api.post<AudioAnalysisResponse[]>(
        '/analyze-batch',
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
          onUploadProgress: (progressEvent) => {
            const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total!);
            console.log('Upload progress:', percentCompleted, '%');
          },
          // Increase timeout for batch processing
          timeout: 300000, // 5 minutes
        }
      )
    , 2, 2000); // 2 retries with 2 second initial delay
    
    console.log('Server batch response:', response.data);
    return response.data;
  } catch (error: any) {
    console.error('API error:', error.response?.data || error.message);
    if (error.code === 'ECONNABORTED') {
      throw new Error('Request timed out. Please try again with fewer files or shorter audio clips.');
    }
    if (error.response?.status === 413) {
      throw new Error('Files are too large. Please try smaller files or fewer files at once.');
    }
    if (error.response?.data?.detail) {
      throw new Error(error.response.data.detail);
    }
    throw new Error('Failed to analyze audio batch. Please try again.');
  }
}; 