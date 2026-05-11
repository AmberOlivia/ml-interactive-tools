export function Landing() {
  return (
    <div className="landing">
      <h1>Machine Learning for Engineers</h1>
      <p className="lead">Interactive tools for exploring how neural networks work.</p>

      <a className="tool-card" href="./forward-prop.html">
        <h2>1 · Forward Propagation →</h2>
        <p>
          Tune inputs, layers, and activations. Watch signals flow through the
          network one layer at a time. Inspect any neuron's weights and
          computation.
        </p>
      </a>

      <a className="tool-card" href="./forward-prop-narrow.html">
        <h2>1b · Forward Propagation (narrow) →</h2>
        <p>
          Single-column layout meant for embedding in Canvas or other narrow
          iframes — same tool, no horizontal scrolling.
        </p>
      </a>

      <a className="tool-card disabled" href="./backprop.html">
        <h2>2 · Backpropagation (coming soon)</h2>
        <p>
          See how gradients flow backward and drive weight updates. Build
          intuition for what makes a network <em>learn</em>.
        </p>
      </a>
    </div>
  );
}
