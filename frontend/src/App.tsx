import { BrowserRouter, Routes, Route } from "react-router-dom";
import Navbar from "./components/Navbar";
import MatchPredictor from "./pages/MatchPredictor";
import BracketSimulator from "./pages/BracketSimulator";
import ModelAccuracy from "./pages/ModelAccuracy";

export default function App() {
  return (
    <BrowserRouter>
      <Navbar />
      <main>
        <Routes>
          <Route path="/" element={<MatchPredictor />} />
          <Route path="/simulate" element={<BracketSimulator />} />
          <Route path="/accuracy" element={<ModelAccuracy />} />
        </Routes>
      </main>
    </BrowserRouter>
  );
}
