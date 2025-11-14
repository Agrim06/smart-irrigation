import Dashboard from './pages/Dashboard.jsx';
import { ThemeProvider } from './contexts/ThemeContext.jsx';

function App() {
  return (
    <ThemeProvider>
      <Dashboard />
    </ThemeProvider>
  );
}

export default App;
