import React from 'react';
import {
  Box,
  VStack,
  Heading,
  Text,
  Progress,
  Accordion,
  AccordionItem,
  AccordionButton,
  AccordionPanel,
  AccordionIcon,
  SimpleGrid,
  Badge,
  Stat,
  StatLabel,
  StatNumber,
  StatHelpText,
  useColorModeValue,
  HStack,
} from '@chakra-ui/react';
import { ChunkStatus } from '../types';

interface AnalysisResultsProps {
  results: any;
}

const formatValue = (value: number, precision: number = 2, unit: string = ''): string => {
  return `${value.toFixed(precision)}${unit ? ' ' + unit : ''}`;
};

interface StatCardProps {
  label: string;
  value: number;
  helpText?: string;
  format?: (value: number) => string;
}

const StatCard = ({ label, value, helpText, format = (v: number) => formatValue(v) }: StatCardProps) => (
  <Stat
    px={4}
    py={2}
    shadow="sm"
    border="1px solid"
    borderColor={useColorModeValue('gray.200', 'gray.700')}
    borderRadius="lg"
    backgroundColor={useColorModeValue('white', 'gray.800')}
  >
    <StatLabel color="gray.500" fontSize="sm">{label}</StatLabel>
    <StatNumber fontSize="lg">{format(value)}</StatNumber>
    {helpText && <StatHelpText fontSize="xs">{helpText}</StatHelpText>}
  </Stat>
);

export const AnalysisResults = ({ results }: AnalysisResultsProps) => {
  const borderColor = useColorModeValue('gray.200', 'gray.700');
  const getStatusColor = (status: ChunkStatus) => {
    switch (status) {
      case ChunkStatus.COMPLETED:
        return 'blackAlpha';
      case ChunkStatus.PROCESSING:
        return 'gray';
      case ChunkStatus.FAILED:
        return 'blackAlpha';
      default:
        return 'gray';
    }
  };

  const getStatusBadgeProps = (status: ChunkStatus) => {
    switch (status) {
      case ChunkStatus.COMPLETED:
        return {
          colorScheme: 'green',
          variant: 'solid',
          children: 'Completed'
        };
      case ChunkStatus.PROCESSING:
        return {
          colorScheme: 'gray',
          variant: 'subtle',
          children: 'Processing'
        };
      case ChunkStatus.FAILED:
        return {
          colorScheme: 'red',
          variant: 'subtle',
          children: 'Failed'
        };
      default:
        return {
          colorScheme: 'gray',
          variant: 'subtle',
          children: 'Pending'
        };
    }
  };

  const progress = results.chunks.filter(
    (chunk: any) => chunk.status === ChunkStatus.COMPLETED
  ).length / results.total_chunks * 100;

  const completedCount = results.chunks.filter((c: any) => c.status === ChunkStatus.COMPLETED).length;
  const pendingCount = results.chunks.filter((c: any) => c.status === ChunkStatus.PENDING).length;
  const processingCount = results.chunks.filter((c: any) => c.status === ChunkStatus.PROCESSING).length;
  const failedCount = results.chunks.filter((c: any) => c.status === ChunkStatus.FAILED).length;

  return (
    <VStack spacing={6} align="stretch" w="100%">
      <Box>
        <Heading size="md" mb={2}>Analysis Progress</Heading>
        <Progress
          value={progress}
          size="lg"
          colorScheme="blackAlpha"
          borderRadius="md"
          hasStripe
          isAnimated={progress < 100}
        />
        <Text mt={2} fontSize="sm" color="gray.500">
          {Math.round(progress)}% Complete ({completedCount} of {results.total_chunks} chunks)
        </Text>
        <HStack mt={1} spacing={4}>
          {completedCount > 0 && <Badge colorScheme="green">Completed: {completedCount}</Badge>}
          {processingCount > 0 && <Badge colorScheme="blue">Processing: {processingCount}</Badge>}
          {pendingCount > 0 && <Badge colorScheme="gray">Pending: {pendingCount}</Badge>}
          {failedCount > 0 && <Badge colorScheme="red">Failed: {failedCount}</Badge>}
        </HStack>
      </Box>

      <Accordion allowMultiple>
        {results.chunks.map((chunk: any) => (
          <AccordionItem
            key={chunk.chunk_id}
            border="1px solid"
            borderColor={borderColor}
            borderRadius="md"
            mb={2}
          >
            <AccordionButton py={3}>
              <Box flex="1" textAlign="left">
                <Text fontWeight="medium">
                  Chunk {chunk.chunk_id + 1}
                  <Badge
                    ml={2}
                    {...getStatusBadgeProps(chunk.status)}
                  >
                    {chunk.status}
                  </Badge>
                </Text>
                <Text fontSize="sm" color="gray.500">
                  {formatValue(chunk.start_time, 1)}s - {formatValue(chunk.end_time, 1)}s
                </Text>
              </Box>
              <AccordionIcon />
            </AccordionButton>

            <AccordionPanel pb={4}>
              {chunk.status === ChunkStatus.COMPLETED && (
                <VStack spacing={6} align="stretch">
                  {/* Transcription */}
                  {chunk.features?.transcription && (
                    <Box>
                      <Heading size="sm" mb={3}>Transcription</Heading>
                      <Box 
                        p={4}
                        bg={useColorModeValue('gray.50', 'gray.900')}
                        borderRadius="md"
                        borderWidth="1px"
                        borderColor={useColorModeValue('gray.200', 'gray.700')}
                      >
                        <Text fontSize="md" color={useColorModeValue('gray.800', 'gray.200')}>
                          {chunk.features.transcription}
                        </Text>
                      </Box>
                    </Box>
                  )}

                  {/* Acoustic Features */}
                  {chunk.features?.acoustic && (
                    <Box>
                      <Heading size="sm" mb={4}>Acoustic Features</Heading>
                      
                      <SimpleGrid columns={[1, 2, 3]} spacing={4} mb={4}>
                        <StatCard
                          label="Pitch"
                          value={chunk.features.acoustic.pitch}
                          helpText="Fundamental frequency"
                          format={(v) => formatValue(v, 1, 'Hz')}
                        />
                        <StatCard
                          label="Energy"
                          value={chunk.features.acoustic.energy}
                          helpText="Root mean square energy"
                          format={(v) => formatValue(v, 4)}
                        />
                        <StatCard
                          label="Zero-Crossing Rate"
                          value={chunk.features.acoustic.zcr}
                          helpText="Signal polarity changes"
                          format={(v) => formatValue(v, 4)}
                        />
                      </SimpleGrid>

                      <Box mb={4}>
                        <Text fontWeight="medium" mb={2}>Spectral Features</Text>
                        <SimpleGrid columns={[1, 2]} spacing={4}>
                          <StatCard
                            label="Spectral Centroid"
                            value={chunk.features.acoustic.spectral.centroid}
                            helpText="Brightness of sound"
                            format={(v) => formatValue(v, 1, 'Hz')}
                          />
                          <StatCard
                            label="Spectral Bandwidth"
                            value={chunk.features.acoustic.spectral.bandwidth}
                            helpText="Width of the spectrum"
                            format={(v) => formatValue(v, 1, 'Hz')}
                          />
                          <StatCard
                            label="Spectral Rolloff"
                            value={chunk.features.acoustic.spectral.rolloff}
                            helpText="Frequency below which 85% of energy is concentrated"
                            format={(v) => formatValue(v, 1, 'Hz')}
                          />
                          <StatCard
                            label="Spectral Flux"
                            value={chunk.features.acoustic.spectral.flux}
                            helpText="Rate of spectral change"
                            format={(v) => formatValue(v, 4)}
                          />
                        </SimpleGrid>
                      </Box>

                      <Box>
                        <Text fontWeight="medium" mb={2}>MFCCs</Text>
                        <Text fontSize="sm" color="gray.600">
                          {chunk.features.acoustic.mfcc.map((v: number) => formatValue(v, 2)).join(', ')}
                        </Text>
                      </Box>
                    </Box>
                  )}

                  {/* Paralinguistic Features */}
                  {chunk.features?.paralinguistic && (
                    <Box>
                      <Heading size="sm" mb={4}>Paralinguistic Features</Heading>
                      
                      <SimpleGrid columns={[1, 2, 3]} spacing={4}>
                        <StatCard
                          label="Pitch Variability"
                          value={chunk.features.paralinguistic.pitch_variability}
                          helpText="Variation in fundamental frequency"
                          format={(v) => formatValue(v, 1, 'Hz')}
                        />
                        <StatCard
                          label="Speech Rate"
                          value={chunk.features.paralinguistic.speech_rate}
                          helpText="Syllables per second"
                          format={(v) => formatValue(v, 2, 'syl/s')}
                        />
                        <StatCard
                          label="Harmonics-to-Noise"
                          value={chunk.features.paralinguistic.hnr}
                          helpText="Voice quality measure"
                          format={(v) => formatValue(v, 1, 'dB')}
                        />
                        <StatCard
                          label="Jitter"
                          value={chunk.features.paralinguistic.jitter}
                          helpText="Frequency variation"
                          format={(v) => formatValue(v, 2, '%')}
                        />
                        <StatCard
                          label="Shimmer"
                          value={chunk.features.paralinguistic.shimmer}
                          helpText="Amplitude variation"
                          format={(v) => formatValue(v, 2, '%')}
                        />
                      </SimpleGrid>
                    </Box>
                  )}
                </VStack>
              )}
              
              {chunk.status === ChunkStatus.FAILED && (
                <Text color="red.500">{chunk.error}</Text>
              )}
              
              {(chunk.status === ChunkStatus.PROCESSING || chunk.status === ChunkStatus.PENDING) && (
                <VStack spacing={4} py={4} align="center">
                  <Progress size="xs" isIndeterminate w="80%" colorScheme="blackAlpha" />
                  <Text fontSize="sm" color="gray.500">
                    {chunk.status === ChunkStatus.PROCESSING ? "Processing..." : "Pending in queue..."}
                  </Text>
                </VStack>
              )}
            </AccordionPanel>
          </AccordionItem>
        ))}
      </Accordion>
    </VStack>
  );
};