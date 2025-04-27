import { ChakraProvider, Box, Container } from '@chakra-ui/react'
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import { AudioAnalyzer } from './components/AudioAnalyzer'
import { JsonViewPage } from './pages/JsonViewPage'
import { theme } from './theme'

function App() {
  return (
    <ChakraProvider theme={theme}>
      <Router>
        <Box minH="100vh" bg="gray.50">
          <Routes>
            <Route path="/" element={
              <Container maxW="container.xl" py={8}>
                <AudioAnalyzer />
              </Container>
            } />
            <Route path="/json-view/:taskId" element={<JsonViewPage />} />
          </Routes>
        </Box>
      </Router>
    </ChakraProvider>
  )
}

export default App
