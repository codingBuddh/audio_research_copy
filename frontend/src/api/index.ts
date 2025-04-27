import axios from 'axios'
import { AudioFeatureType, AudioAnalysisResponse } from '../types'

const API_BASE_URL = 'http://localhost:8000/api/v1'

// Create axios instance with default config
export const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000, // 30 second timeout
  headers: {
    'Accept': 'application/json',
  }
});

export const analyzeAudio = async (
  file: File,
  featureTypes: AudioFeatureType[]
): Promise<AudioAnalysisResponse> => {
  const formData = new FormData()
  formData.append('file', file)
  
  // Convert feature types to strings and log the data
  const featureTypesStr = JSON.stringify(featureTypes.map(ft => ft.toString()))
  console.log('Sending feature types:', featureTypesStr)
  formData.append('feature_types', featureTypesStr)
  formData.append('chunk_duration', '60.0')

  try {
    const response = await api.post<AudioAnalysisResponse>(
      '/analyze',
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (progressEvent) => {
          const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total!);
          console.log('Upload progress:', percentCompleted, '%');
        },
      }
    )
    console.log('Server response:', response.data)
    return response.data
  } catch (error: any) {
    console.error('API error:', error.response?.data || error.message)
    if (error.code === 'ECONNABORTED') {
      throw new Error('Request timed out. Please try again.');
    }
    if (error.response?.status === 413) {
      throw new Error('File is too large. Please try a smaller file.');
    }
    if (error.response?.data?.detail) {
      throw new Error(error.response.data.detail);
    }
    throw new Error('Failed to analyze audio. Please try again.');
  }
}

export const connectToWebSocket = (taskId: string): WebSocket => {
  const ws = new WebSocket(`ws://localhost:8000/api/v1/ws/${taskId}`)
  
  ws.onopen = () => {
    console.log('WebSocket connected')
  }
  
  ws.onerror = (error) => {
    console.error('WebSocket error:', error)
  }
  
  ws.onclose = (event) => {
    console.log('WebSocket closed:', event.code, event.reason)
  }
  
  return ws
}

export const getTaskStatus = async (taskId: string): Promise<AudioAnalysisResponse> => {
  try {
    const response = await api.get<AudioAnalysisResponse>(`/status/${taskId}`)
    return response.data
  } catch (error: any) {
    console.error('Error getting task status:', error.response?.data || error.message)
    throw new Error('Failed to get analysis status. Please try again.');
  }
} 