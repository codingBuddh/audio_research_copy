import React, { useEffect, useState } from 'react';
import {
  Box,
  Container,
  Heading,
  Text,
  VStack,
  useColorModeValue,
  Tabs,
  TabList,
  TabPanels,
  Tab,
  TabPanel,
  Skeleton,
} from '@chakra-ui/react';
import { JsonViewer } from '../components/JsonViewer/JsonViewer';

interface JsonViewPageProps {
  taskId?: string;
}

export const JsonViewPage: React.FC<JsonViewPageProps> = ({ taskId }) => {
  const [analysisData, setAnalysisData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      if (!taskId) {
        setLoading(false);
        return;
      }

      try {
        const response = await fetch(`/api/v1/audio/chunks/${taskId}`);
        if (!response.ok) {
          throw new Error('Failed to fetch analysis data');
        }
        const data = await response.json();
        setAnalysisData(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'An error occurred');
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [taskId]);

  const bgColor = useColorModeValue('white', 'gray.800');
  const borderColor = useColorModeValue('gray.200', 'gray.700');

  return (
    <Container maxW="container.xl" py={8}>
      <VStack spacing={8} align="stretch">
        <Box>
          <Heading size="xl" mb={2}>
            Analysis Results
          </Heading>
          <Text color={useColorModeValue('gray.600', 'gray.400')}>
            View and download the complete analysis data in JSON format
          </Text>
        </Box>

        <Box
          bg={bgColor}
          p={6}
          borderRadius="xl"
          border="1px"
          borderColor={borderColor}
          shadow="sm"
        >
          {loading ? (
            <VStack spacing={4} align="stretch">
              <Skeleton height="40px" />
              <Skeleton height="400px" />
            </VStack>
          ) : error ? (
            <Text color="red.500">{error}</Text>
          ) : !analysisData ? (
            <Text>No analysis data available</Text>
          ) : (
            <Tabs variant="enclosed">
              <TabList>
                <Tab>Complete Data</Tab>
                <Tab>Chunk-wise View</Tab>
              </TabList>

              <TabPanels>
                <TabPanel>
                  <JsonViewer
                    data={analysisData}
                    title="Complete Analysis Results"
                  />
                </TabPanel>
                <TabPanel>
                  <VStack spacing={6} align="stretch">
                    {analysisData.chunks?.map((chunk: any, index: number) => (
                      <Box key={chunk.chunk_id}>
                        <JsonViewer
                          data={chunk}
                          title={`Chunk ${index + 1} (${chunk.start_time.toFixed(2)}s - ${chunk.end_time.toFixed(2)}s)`}
                        />
                      </Box>
                    ))}
                  </VStack>
                </TabPanel>
              </TabPanels>
            </Tabs>
          )}
        </Box>
      </VStack>
    </Container>
  );
}; 