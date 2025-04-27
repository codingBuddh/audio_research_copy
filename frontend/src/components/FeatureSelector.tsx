import React from 'react';
import {
  VStack,
  Box,
  Heading,
  Text,
  Button,
  SimpleGrid,
  useColorModeValue,
  Icon,
  Flex,
} from '@chakra-ui/react';
import { AudioFeatureType } from '../types';
import { MdAudiotrack, MdSentimentSatisfiedAlt, MdTranscribe, MdGraphicEq } from 'react-icons/md';

export const FEATURE_OPTIONS = [
  {
    type: AudioFeatureType.TRANSCRIPTION,
    label: 'Transcription',
    icon: MdTranscribe,
    description: 'Convert speech to text using Whisper AI',
    features: [
      'High-accuracy speech recognition',
      'Support for multiple languages',
      'Punctuation and formatting',
    ],
  },
  {
    type: AudioFeatureType.ACOUSTIC,
    label: 'Acoustic Features',
    icon: MdGraphicEq,
    description: 'Extract fundamental acoustic properties including MFCCs, pitch, formants, and spectral features',
    features: [
      'Mel-frequency cepstral coefficients (MFCCs)',
      'Fundamental frequency (pitch)',
      'Formant frequencies',
      'Energy and zero-crossing rate',
      'Spectral characteristics',
    ],
  },
  {
    type: AudioFeatureType.PARALINGUISTIC,
    label: 'Paralinguistic Features',
    icon: MdSentimentSatisfiedAlt,
    description: 'Analyze voice quality and emotional characteristics of speech',
    features: [
      'Pitch variability',
      'Speech rate analysis',
      'Voice quality (jitter/shimmer)',
      'Harmonics-to-noise ratio',
      'Emotional markers',
    ],
  },
];

interface FeatureSelectorProps {
  selectedFeatures: AudioFeatureType[];
  onFeaturesChange: (features: AudioFeatureType[]) => void;
  isAnalyzing: boolean;
  onAnalyze: () => void;
  hideAnalyzeButton?: boolean;
}

export const FeatureSelector: React.FC<FeatureSelectorProps> = ({
  selectedFeatures,
  onFeaturesChange,
  isAnalyzing,
  onAnalyze,
  hideAnalyzeButton = false,
}) => {
  const bgColor = useColorModeValue('white', 'black');
  const borderColor = useColorModeValue('gray.200', 'gray.800');
  const selectedBg = useColorModeValue('gray.100', 'gray.900');
  const hoverBg = useColorModeValue('gray.50', 'gray.800');

  return (
    <VStack spacing={6} align="stretch" w="100%" p={4}>
      <Box>
        <Heading size="md" mb={4}>Psychometric Audio Analysis</Heading>
        <Text mb={6} color={useColorModeValue('gray.700', 'gray.300')}>
          Select the features you want to extract from your audio file.
        </Text>
        
        <SimpleGrid columns={[1, null, 2]} spacing={4} mb={8}>
          {FEATURE_OPTIONS.map((feature) => (
            <Box
              key={feature.type}
              p={4}
              borderWidth="1px"
              borderRadius="lg"
              borderColor={borderColor}
              bg={selectedFeatures.includes(feature.type) ? selectedBg : bgColor}
              cursor="pointer"
              onClick={() => {
                const newFeatures = selectedFeatures.includes(feature.type)
                  ? selectedFeatures.filter(f => f !== feature.type)
                  : [...selectedFeatures, feature.type];
                onFeaturesChange(newFeatures);
              }}
              _hover={{
                bg: selectedFeatures.includes(feature.type) ? selectedBg : hoverBg,
                borderColor: 'black',
              }}
              transition="all 0.2s"
            >
              <Flex align="center" mb={2}>
                <Icon
                  as={feature.icon}
                  boxSize={6}
                  color={selectedFeatures.includes(feature.type) ? 'black' : 'gray.500'}
                  mr={2}
                />
                <Heading size="sm" color={useColorModeValue('black', 'white')}>{feature.label}</Heading>
              </Flex>
              
              <Text fontSize="sm" color={useColorModeValue('gray.700', 'gray.300')} mb={3}>
                {feature.description}
              </Text>
              
              <VStack align="start" spacing={1}>
                {feature.features.map((item, index) => (
                  <Text
                    key={`${feature.type}-${index}`}
                    fontSize="xs"
                    color={useColorModeValue('gray.600', 'gray.400')}
                    pl={4}
                    position="relative"
                    _before={{
                      content: '"•"',
                      position: "absolute",
                      left: 1,
                      color: useColorModeValue('gray.600', 'gray.400')
                    }}
                  >
                    {item}
                  </Text>
                ))}
              </VStack>
            </Box>
          ))}
        </SimpleGrid>
      </Box>
      
      {!hideAnalyzeButton && (
        <Button
          colorScheme={selectedFeatures.length > 0 ? "blackAlpha" : "gray"}
          size="lg"
          isDisabled={selectedFeatures.length === 0 || isAnalyzing}
          isLoading={isAnalyzing}
          loadingText="Analyzing..."
          onClick={onAnalyze}
          w="100%"
          bg={selectedFeatures.length > 0 ? "black" : "gray.200"}
          color="white"
          _hover={{
            bg: selectedFeatures.length > 0 ? "gray.800" : "gray.200",
          }}
          _active={{
            bg: selectedFeatures.length > 0 ? "gray.700" : "gray.200",
          }}
        >
          Analyze Audio
        </Button>
      )}
    </VStack>
  );
}; 