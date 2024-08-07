import { Routes, Route } from 'react-router-dom'

import Barra from './components/Barra'
import ErrorPage from './pages/errorPage'

import Submission from './pages/Submission'
import Output from './pages/Output'

function App() {

  return (
    <>
      <Barra>
        <Routes>
          <Route path="/" element={<Submission />} />
          <Route path="/Output" element={<Output />} />
          <Route path="/error" element={<ErrorPage />} />
          <Route path="*" element={<ErrorPage />} />
        </Routes>
      </Barra>
      
    </>
  )
}

export default App
