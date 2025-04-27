import React, { useState, useCallback, useEffect } from 'react';
import {
  Box,
  VStack,
  Heading,
  Text,
  Button,
  useToast,
  Checkbox,
  Divider,
  Table,
  Thead,
  Tbody,
  Tr,
  Th,
  Td,
  Badge,
  Progress,
  HStack,
  Switch,
  FormControl,
  FormLabel,
  IconButton,
  useDisclosure,
  Modal,
  ModalOverlay,
  ModalContent,
  ModalHeader,
  ModalBody,
  ModalCloseButton,
  ModalFooter,
  useColorModeValue,
  Tooltip,
} from '@chakra-ui/react';
import { FileUploader } from './FileUploader';
import { FeatureSelector } from './FeatureSelector';
import { AudioFeatureType, AudioAnalysisResponse, ChunkStatus } from '../types';
import { analyzeAudioBatch } from '../api/batch';
import { DownloadIcon, ViewIcon } from '@chakra-ui/icons';
import { AnalysisResults } from './AnalysisResults';
import { getTaskStatus } from '../api';

interface BatchProcessorProps {
  selectedFeatures: AudioFeatureType[];
  onFeaturesChange: (features: AudioFeatureType[]) => void;
}

interface ResultStatus {
  isError: boolean;
  message: string;
}

export const BatchProcessor: React.FC<BatchProcessorProps> = ({
  selectedFeatures,
  onFeaturesChange,
}) => {
  const [files, setFiles] = useState<File[]>([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [results, setResults] = useState<AudioAnalysisResponse[]>([]);
  const [waitForCompletion, setWaitForCompletion] = useState(false);
  const [resultStatus, setResultStatus] = useState<ResultStatus | null>(null);
  const toast = useToast();
  const { isOpen, onOpen, onClose } = useDisclosure();
  const [selectedResult, setSelectedResult] = useState<AudioAnalysisResponse | null>(null);
  const [pollingInterval, setPollingInterval] = useState<NodeJS.Timeout | null>(null);

  // Set up polling for task updates
  useEffect(() => {
    // Only poll if we have results with PENDING status
    const hasPendingTasks = results.some(result => 
      result.chunks.some(chunk => 
        chunk.status === ChunkStatus.PENDING || chunk.status === ChunkStatus.PROCESSING
      )
    );

    if (results.length > 0 && hasPendingTasks) {
      console.log('Setting up polling for batch tasks');
      
      // Clear any existing interval first
      if (pollingInterval) {
        clearInterval(pollingInterval);
      }
      
      // Set up new polling interval (every 3 seconds)
      const interval = setInterval(async () => {
        try {
          let updatedResults = [...results];
          let hasUpdates = false;
          
          // For each task, check if it needs to be updated
          for (let i = 0; i < updatedResults.length; i++) {
            const result = updatedResults[i];
            
            // Only refresh tasks that have PENDING/PROCESSING chunks
            if (result.chunks.some(chunk => 
              chunk.status === ChunkStatus.PENDING || chunk.status === ChunkStatus.PROCESSING
            )) {
              try {
                console.log(`Polling task ${result.task_id}`);
                const updatedTask = await getTaskStatus(result.task_id);
                if (JSON.stringify(updatedTask) !== JSON.stringify(result)) {
                  updatedResults[i] = updatedTask;
                  hasUpdates = true;
                }
              } catch (error) {
                console.error(`Error polling task ${result.task_id}:`, error);
              }
            }
          }
          
          // Only update state if there were changes
          if (hasUpdates) {
            console.log('Updating results with polled data');
            setResults([...updatedResults]);
          }
          
          // Check if we should keep polling
          const stillHasPendingTasks = updatedResults.some(result => 
            result.chunks.some(chunk => 
              chunk.status === ChunkStatus.PENDING || chunk.status === ChunkStatus.PROCESSING
            )
          );
          
          if (!stillHasPendingTasks) {
            console.log('No more pending tasks, stopping polling');
            clearInterval(interval);
            setPollingInterval(null);
          }
        } catch (error) {
          console.error('Error during task polling:', error);
        }
      }, 3000);
      
      setPollingInterval(interval);
      
      // Cleanup on unmount
      return () => {
        if (interval) {
          clearInterval(interval);
        }
      };
    }
  }, [results]);
  
  // Additional cleanup when component unmounts
  useEffect(() => {
    return () => {
      if (pollingInterval) {
        clearInterval(pollingInterval);
      }
    };
  }, [pollingInterval]);

  const handleFilesSelect = useCallback((selectedFiles: File[]) => {
    setFiles(selectedFiles);
  }, []);

  const handleAnalyze = async () => {
    if (files.length === 0 || selectedFeatures.length === 0) {
      toast({
        title: 'Error',
        description: 'Please select at least one file and feature type',
        status: 'error',
        duration: 3000,
      });
      return;
    }

    setIsAnalyzing(true);
    setResultStatus(null);
    
    try {
      // Set a longer timeout for larger batches
      const timeout = Math.max(120000, files.length * 30000); // At least 2 minutes, or 30s per file
      
      toast({
        title: 'Processing',
        description: `Starting analysis of ${files.length} files...`,
        status: 'info',
        duration: 3000,
      });
      
      const batchResults = await analyzeAudioBatch(files, selectedFeatures, waitForCompletion);
      setResults(batchResults);
      
      // Check if any files failed
      const failedFiles = batchResults.filter(result => 
        result.chunks.some(chunk => chunk.status === ChunkStatus.FAILED)
      );
      
      if (failedFiles.length > 0) {
        const errorMessage = `${failedFiles.length} of ${batchResults.length} files had processing errors.`;
        setResultStatus({
          isError: true,
          message: errorMessage
        });
        
        toast({
          title: 'Partial Success',
          description: errorMessage,
          status: 'warning',
          duration: 5000,
        });
      } else {
        setResultStatus({
          isError: false,
          message: `Successfully started processing ${batchResults.length} files`
        });
        
        toast({
          title: 'Success',
          description: `Started processing ${batchResults.length} audio files`,
          status: 'success',
          duration: 3000,
        });
      }
    } catch (error: any) {
      setResultStatus({
        isError: true,
        message: error.message || 'Failed to analyze audio files'
      });
      
      toast({
        title: 'Error',
        description: error.message || 'Failed to analyze audio files',
        status: 'error',
        duration: 5000,
      });
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleViewResult = (result: AudioAnalysisResponse) => {
    setSelectedResult(result);
    onOpen();
  };

  const calculateProgress = (result: AudioAnalysisResponse) => {
    const completed = result.chunks.filter(chunk => chunk.status === ChunkStatus.COMPLETED).length;
    return (completed / result.total_chunks) * 100;
  };

  const handleDownloadJSON = useCallback(() => {
    if (results.length === 0) return;

    // Create a formatted JSON string
    const jsonString = JSON.stringify(results, null, 2);
    
    // Create a blob from the JSON string
    const blob = new Blob([jsonString], { type: 'application/json' });
    
    // Create a URL for the blob
    const url = URL.createObjectURL(blob);
    
    // Create a temporary link element
    const link = document.createElement('a');
    link.href = url;
    link.download = `audio_batch_analysis_${new Date().toISOString()}.json`;
    
    // Append link to body, click it, and remove it
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    
    // Clean up the URL
    URL.revokeObjectURL(url);

    toast({
      title: 'Success',
      description: 'Batch analysis results downloaded as JSON',
      status: 'success',
      duration: 3000,
    });
  }, [results, toast]);

  return (
    <VStack spacing={8} align="stretch" w="100%">
      <Box
        bg={useColorModeValue('white', 'gray.800')}
        p={8}
        borderRadius="xl"
        shadow="lg"
        border="1px solid"
        borderColor={useColorModeValue('gray.100', 'gray.700')}
      >
        <VStack spacing={6} align="stretch">
          <Heading size="lg">Batch Audio Analysis</Heading>
          <Text>Select multiple audio files to analyze in sequence and combine results into a single output.</Text>
          
          <FileUploader 
            onFileSelect={() => {}} 
            onMultipleFilesSelect={handleFilesSelect}
            allowMultiple={true}
          />
          
          <Divider />
          
          <Box>
            <Heading size="md" mb={4}>Feature Selection</Heading>
            <FeatureSelector
              selectedFeatures={selectedFeatures}
              onFeaturesChange={onFeaturesChange}
              onAnalyze={() => {}}
              isAnalyzing={false}
              hideAnalyzeButton
            />
          </Box>
          
          <FormControl display="flex" alignItems="center">
            <FormLabel htmlFor="wait-for-completion" mb="0">
              Wait for processing to complete before returning results
            </FormLabel>
            <Switch 
              id="wait-for-completion" 
              isChecked={waitForCompletion}
              onChange={() => setWaitForCompletion(!waitForCompletion)}
              colorScheme="blackAlpha"
            />
          </FormControl>
          
          <Button
            colorScheme="blackAlpha"
            size="lg"
            onClick={handleAnalyze}
            isLoading={isAnalyzing}
            loadingText="Processing Files"
            isDisabled={files.length === 0 || selectedFeatures.length === 0}
          >
            Analyze {files.length} Audio File{files.length !== 1 ? 's' : ''}
          </Button>
        </VStack>
      </Box>
      
      {results.length > 0 && (
        <Box
          bg={useColorModeValue('white', 'gray.800')}
          p={8}
          borderRadius="xl"
          shadow="lg"
          border="1px solid"
          borderColor={useColorModeValue('gray.100', 'gray.700')}
        >
          <VStack spacing={6} align="stretch">
            <HStack justify="space-between">
              <Heading size="lg">Batch Analysis Results</Heading>
              <Button
                leftIcon={<DownloadIcon />}
                onClick={handleDownloadJSON}
                colorScheme="blackAlpha"
                variant="outline"
                isDisabled={results.length === 0}
              >
                Download JSON
              </Button>
            </HStack>
            
            {resultStatus && (
              <Box 
                p={3} 
                bg={resultStatus.isError ? 'red.50' : 'green.50'} 
                color={resultStatus.isError ? 'red.800' : 'green.800'}
                borderRadius="md"
                borderWidth="1px"
                borderColor={resultStatus.isError ? 'red.200' : 'green.200'}
              >
                <Text fontWeight="medium">{resultStatus.message}</Text>
                {resultStatus.isError && (
                  <Text fontSize="sm" mt={1}>
                    Some files may be stuck in PENDING state. Try reducing the file size or selecting fewer files at once.
                  </Text>
                )}
              </Box>
            )}
            
            <Table variant="simple">
              <Thead>
                <Tr>
                  <Th>File Name</Th>
                  <Th>Status</Th>
                  <Th>Progress</Th>
                  <Th>Actions</Th>
                </Tr>
              </Thead>
              <Tbody>
                {results.map((result, idx) => {
                  const progress = calculateProgress(result);
                  const hasError = result.chunks.some(chunk => chunk.status === ChunkStatus.FAILED);
                  
                  return (
                    <Tr key={result.task_id}>
                      <Td>
                        {result.original_filename || `File ${idx + 1}`}
                      </Td>
                      <Td>
                        <Badge
                          colorScheme={
                            hasError
                              ? 'red'
                              : progress === 100
                              ? 'green'
                              : progress > 0
                              ? 'orange'
                              : 'gray'
                          }
                        >
                          {hasError
                            ? 'Error'
                            : progress === 100
                            ? 'Completed'
                            : progress > 0
                            ? 'Processing'
                            : 'Pending'}
                        </Badge>
                      </Td>
                      <Td width="30%">
                        <Tooltip 
                          label={hasError ? 'Processing encountered errors' : `${progress.toFixed(0)}% complete`}
                          aria-label="Progress tooltip"
                        >
                          <Progress
                            value={progress}
                            size="sm"
                            colorScheme={hasError ? "red" : "blackAlpha"}
                            borderRadius="md"
                          />
                        </Tooltip>
                      </Td>
                      <Td>
                        <IconButton
                          aria-label="View results"
                          icon={<ViewIcon />}
                          onClick={() => handleViewResult(result)}
                          variant="ghost"
                        />
                      </Td>
                    </Tr>
                  );
                })}
              </Tbody>
            </Table>
          </VStack>
        </Box>
      )}
      
      <Modal isOpen={isOpen} onClose={onClose} size="5xl" scrollBehavior="inside">
        <ModalOverlay />
        <ModalContent>
          <ModalHeader>
            Analysis Results: {selectedResult?.original_filename}
          </ModalHeader>
          <ModalCloseButton />
          <ModalBody>
            {selectedResult && <AnalysisResults results={selectedResult} />}
          </ModalBody>
          <ModalFooter>
            <Button onClick={onClose}>Close</Button>
          </ModalFooter>
        </ModalContent>
      </Modal>
    </VStack>
  );
}; 