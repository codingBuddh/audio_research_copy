import React from 'react';
import {
  Box,
  Button,
  useColorModeValue,
  Text,
  VStack,
  HStack,
  useToast,
  IconButton,
  Tooltip,
} from '@chakra-ui/react';
import { DownloadIcon, CopyIcon } from '@chakra-ui/icons';

interface JsonViewerProps {
  data: any;
  title?: string;
}

export const JsonViewer: React.FC<JsonViewerProps> = ({ data, title }) => {
  const toast = useToast();
  const bgColor = useColorModeValue('gray.50', 'gray.900');
  const borderColor = useColorModeValue('gray.200', 'gray.700');
  const textColor = useColorModeValue('gray.800', 'gray.100');

  const handleDownload = () => {
    try {
      const jsonString = JSON.stringify(data, null, 2);
      const blob = new Blob([jsonString], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `audio_analysis_${new Date().toISOString()}.json`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);

      toast({
        title: 'Success',
        description: 'JSON file downloaded successfully',
        status: 'success',
        duration: 3000,
      });
    } catch (error) {
      toast({
        title: 'Error',
        description: 'Failed to download JSON file',
        status: 'error',
        duration: 3000,
      });
    }
  };

  const handleCopy = () => {
    try {
      const jsonString = JSON.stringify(data, null, 2);
      navigator.clipboard.writeText(jsonString);
      toast({
        title: 'Success',
        description: 'JSON copied to clipboard',
        status: 'success',
        duration: 2000,
      });
    } catch (error) {
      toast({
        title: 'Error',
        description: 'Failed to copy JSON',
        status: 'error',
        duration: 2000,
      });
    }
  };

  return (
    <VStack spacing={4} align="stretch" width="100%">
      <HStack justify="space-between" align="center">
        {title && (
          <Text fontSize="xl" fontWeight="bold">
            {title}
          </Text>
        )}
        <HStack spacing={2}>
          <Tooltip label="Copy to clipboard">
            <IconButton
              aria-label="Copy JSON"
              icon={<CopyIcon />}
              onClick={handleCopy}
              size="sm"
            />
          </Tooltip>
          <Button
            leftIcon={<DownloadIcon />}
            colorScheme="blue"
            onClick={handleDownload}
            size="sm"
          >
            Download JSON
          </Button>
        </HStack>
      </HStack>

      <Box
        p={4}
        bg={bgColor}
        borderRadius="md"
        border="1px"
        borderColor={borderColor}
        overflowX="auto"
      >
        <pre
          style={{
            margin: 0,
            color: textColor,
            fontSize: '14px',
            lineHeight: '1.5',
            fontFamily: 'monospace',
          }}
        >
          {JSON.stringify(data, null, 2)}
        </pre>
      </Box>
    </VStack>
  );
}; 