import { useEffect, useRef, useState } from 'react'
import { Box, IconButton, HStack, Text } from '@chakra-ui/react'
import { FiPlay, FiPause } from 'react-icons/fi'
import WaveSurfer from 'wavesurfer.js'

interface AudioVisualizerProps {
  file: File
  results: any
  progress: number
}

export const AudioVisualizer = ({ file, results, progress }: AudioVisualizerProps) => {
  const containerRef = useRef<HTMLDivElement>(null)
  const wavesurferRef = useRef<WaveSurfer | null>(null)
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)

  useEffect(() => {
    if (containerRef.current && file) {
      // Initialize WaveSurfer
      const wavesurfer = WaveSurfer.create({
        container: containerRef.current,
        waveColor: '#3182ce',
        progressColor: '#2b6cb0',
        cursorColor: '#2c5282',
        height: 100,
        normalize: true,
        responsive: true,
      })

      // Load audio file
      wavesurfer.loadBlob(file)

      // Add event listeners
      wavesurfer.on('play', () => setIsPlaying(true))
      wavesurfer.on('pause', () => setIsPlaying(false))
      wavesurfer.on('audioprocess', (time) => setCurrentTime(time))

      wavesurferRef.current = wavesurfer

      return () => {
        wavesurfer.destroy()
      }
    }
  }, [file])

  const togglePlayPause = () => {
    if (wavesurferRef.current) {
      if (isPlaying) {
        wavesurferRef.current.pause()
      } else {
        wavesurferRef.current.play()
      }
    }
  }

  const formatTime = (time: number) => {
    const minutes = Math.floor(time / 60)
    const seconds = Math.floor(time % 60)
    return `${minutes}:${seconds.toString().padStart(2, '0')}`
  }

  return (
    <Box mt={6}>
      <Box ref={containerRef} />
      <HStack mt={4} spacing={4} justify="center">
        <IconButton
          aria-label={isPlaying ? 'Pause' : 'Play'}
          icon={isPlaying ? <FiPause /> : <FiPlay />}
          onClick={togglePlayPause}
          colorScheme="brand"
          size="lg"
          isRound
        />
        <Text color="gray.600">
          {formatTime(currentTime)}
        </Text>
      </HStack>
    </Box>
  )
} 