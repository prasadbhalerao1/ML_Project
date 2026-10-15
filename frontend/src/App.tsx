import { Navigate, Route, Routes } from 'react-router-dom';
import { Layout } from './components/Layout';
import { PredictionProvider } from './hooks/usePrediction';
import { DashboardPage } from './pages/DashboardPage';
import { ModelInfoPage } from './pages/ModelInfoPage';
import { PredictPage } from './pages/PredictPage';
import { ResultPage } from './pages/ResultPage';

export default function App() {
  return (
    <PredictionProvider>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<DashboardPage />} />
          <Route path="predict" element={<PredictPage />} />
          <Route path="result" element={<ResultPage />} />
          <Route path="model" element={<ModelInfoPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </PredictionProvider>
  );
}
