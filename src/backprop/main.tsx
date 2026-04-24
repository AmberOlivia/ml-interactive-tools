import React from 'react';
import ReactDOM from 'react-dom/client';
import '../shared/ui/styles.css';

function Placeholder() {
  return (
    <div className="landing">
      <h1>Backpropagation</h1>
      <p className="lead">
        Coming soon. This tool will visualize gradient flow and weight updates.
      </p>
      <a className="tool-card" href="./index.html">
        <h2>← Back to all tools</h2>
        <p>Return to the tool index.</p>
      </a>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <Placeholder />
  </React.StrictMode>,
);
