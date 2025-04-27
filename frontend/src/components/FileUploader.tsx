import React, { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import {
  Box,
  Text,
  Icon,
  VStack,
  Button,
  useColorModeValue,
  List,
  ListItem,
  HStack,
  Spacer,
  IconButton,
  Badge,
} from '@chakra-ui/react'
import { FiUpload, FiFile, FiX } from 'react-icons/fi'

interface FileUploaderProps {
  onFileSelect: (file: File) => void
  onMultipleFilesSelect?: (files: File[]) => void
  allowMultiple?: boolean
}

export const FileUploader: React.FC<FileUploaderProps> = ({ 
  onFileSelect, 
  onMultipleFilesSelect,
  allowMultiple = false
}) => {
  const [selectedFiles, setSelectedFiles] = useState<File[]>([])
  
  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      if (allowMultiple) {
        setSelectedFiles(prev => [...prev, ...acceptedFiles])
        if (onMultipleFilesSelect) {
          onMultipleFilesSelect([...selectedFiles, ...acceptedFiles])
        }
      } else {
        setSelectedFiles([acceptedFiles[0]])
        onFileSelect(acceptedFiles[0])
      }
    }
  }, [onFileSelect, onMultipleFilesSelect, allowMultiple, selectedFiles])

  const removeFile = (index: number) => {
    const newFiles = [...selectedFiles]
    newFiles.splice(index, 1)
    setSelectedFiles(newFiles)
    
    if (allowMultiple && onMultipleFilesSelect) {
      onMultipleFilesSelect(newFiles)
    } else if (newFiles.length > 0) {
      onFileSelect(newFiles[0])
    }
  }

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'audio/*': ['.mp3', '.wav', '.m4a', '.aac', '.ogg', '.flac'],
    },
    multiple: allowMultiple,
  })

  return (
    <VStack spacing={4} width="100%">
      <Box
        {...getRootProps()}
        p={6}
        borderWidth={2}
        borderRadius="lg"
        borderStyle="dashed"
        borderColor={useColorModeValue(
          isDragActive ? 'blackAlpha.500' : 'blackAlpha.300',
          isDragActive ? 'whiteAlpha.500' : 'whiteAlpha.300'
        )}
        backgroundColor={useColorModeValue(
          isDragActive ? 'blackAlpha.50' : 'transparent',
          isDragActive ? 'whiteAlpha.50' : 'transparent'
        )}
        cursor="pointer"
        transition="all 0.2s"
        _hover={{
          borderColor: 'blackAlpha.500',
          backgroundColor: useColorModeValue('blackAlpha.50', 'whiteAlpha.50'),
        }}
        width="100%"
      >
        <input {...getInputProps()} />
        <VStack spacing={3}>
          <Icon as={FiUpload} w={10} h={10} color="gray.500" />
          <Text fontWeight="medium" textAlign="center">
            {isDragActive
              ? 'Drop the audio files here'
              : 'Drag & drop audio files here or click to browse'}
          </Text>
          <Text fontSize="sm" color="gray.500" textAlign="center">
            Supported formats: MP3, WAV, M4A, AAC, OGG, FLAC
          </Text>
          {allowMultiple && (
            <Badge colorScheme="black" variant="outline" mt={2}>
              Multiple Files Allowed
            </Badge>
          )}
        </VStack>
      </Box>

      {selectedFiles.length > 0 && (
        <List spacing={2} width="100%">
          {selectedFiles.map((file, index) => (
            <ListItem
              key={`${file.name}-${index}`}
              p={2}
              borderWidth={1}
              borderRadius="md"
              borderColor={useColorModeValue('gray.200', 'gray.700')}
            >
              <HStack>
                <Icon as={FiFile} color="gray.500" />
                <Text fontSize="sm" noOfLines={1}>
                  {file.name}
                </Text>
                <Text fontSize="xs" color="gray.500">
                  ({(file.size / (1024 * 1024)).toFixed(2)} MB)
                </Text>
                <Spacer />
                <IconButton
                  aria-label="Remove file"
                  icon={<FiX />}
                  size="sm"
                  variant="ghost"
                  onClick={() => removeFile(index)}
                />
              </HStack>
            </ListItem>
          ))}
        </List>
      )}
    </VStack>
  )
} 