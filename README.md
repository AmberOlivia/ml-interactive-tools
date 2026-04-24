# ML for Engineers — Interactive Tools

Browser-based interactive tools for teaching neural network concepts. Inspired by [TensorFlow Playground](https://playground.tensorflow.org/), but focused on **step-by-step pedagogy** of forward and backward propagation.

## Tools

1. **Forward Propagation** (`/forward-prop.html`) — tune inputs, layers, activations; step through the forward pass one layer at a time.
2. **Backpropagation** (`/backprop.html`) — *coming soon*.

## Develop

```sh
npm install
npm run dev
```

Open <http://localhost:5173/> for the landing page. The two tools are at `/forward-prop.html` and `/backprop.html`.

## Build

```sh
npm run build
npm run preview
```

## Deploy to GitHub Pages

1. Push this repo to GitHub.
2. In the repo settings → Pages, set **Source** to "GitHub Actions".
3. Push to `main` — the workflow at `.github/workflows/deploy.yml` builds and deploys automatically.

The site will be served at `https://<user>.github.io/<repo>/`. The workflow sets `VITE_BASE` to the repo subpath so asset URLs resolve correctly.

## Embed in Canvas

Add an iframe to a Canvas page:

```html
<iframe src="https://<user>.github.io/<repo>/forward-prop.html"
        width="100%" height="900"
        style="border: 0"></iframe>
```

## Layout

```
src/
  landing/       — tool index page
  forward-prop/  — forward-propagation tool
  backprop/      — backprop tool (placeholder)
  shared/
    nn/          — network, activations, features, rng
    datasets/    — circle, xor, gaussian, spiral
    viz/         — NetworkGraph, OutputHeatmap, colors
    ui/          — shared styles, Slider
```
