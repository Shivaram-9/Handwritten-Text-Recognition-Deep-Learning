# Background Redesign Report

## Old Background Approach
The previous design relied on a single fixed `<div class="bg-wrapper">` containing generic blurred CSS gradients (`.mesh-blob`). This background remained static across all sections and did not communicate the specific nature of a Handwritten Text Recognition (HTR) system. It felt generic and loosely themed around basic AI/startup templates.

## New Background Concept
The frontend architecture was structurally overhauled to support a **Section-Based Environment**. 
Each section now visually transitions the user through an "AI Handwriting Laboratory":
- **Hero**: Neural network nodes representing deep learning.
- **Image Upload**: Document paper texture with subtle cursive handwriting strokes.
- **Computer Vision Pipeline**: OCR grid lines and scanning lasers.
- **Inference Metrics & History**: Archival/technical environments representing data storage and performance tracking.

## Light Mode vs. Dark Mode Aesthetics
The colors were tightly calibrated via updated CSS variables:
- **Light Mode**: Mimics physical paper and academia. Uses off-white/warm neutral bases (`#F8F9FA`) with very faint ink artifacts and soft blue technical lighting.
- **Dark Mode**: Mimics advanced digital research. Uses deep navy bases (`#0B1220`) with glowing electric-cyan (`#06b6d4`) and purple (`#8B5CF6`) pathways acting as illuminated ink.

## Animation & Performance
Animations were exclusively built using GPU-accelerated CSS properties (`opacity`, `transform`) and SVG attributes (`stroke-dashoffset`). 
- **Scanning Beam**: Vertical linear-gradient animation representing OCR scanning.
- **Draw Stroke**: SVG path animation mimicking real-time cursive handwriting generation.
- **Pulsing Nodes**: Subtle opacity scaling on neural circles.
- **Performance**: No expensive JavaScript `requestAnimationFrame` loops were used. 
- **Accessibility**: All animations are strictly wrapped in `@media (prefers-reduced-motion: no-preference)` to respect user OS settings.

## Structural Integrity
The `max-width: 1000px` content constraint was successfully preserved by introducing full-width `.page-section` wrappers that house inner `.content-wrapper` elements. This ensured all existing JavaScript element queries (`getElementById`) remained fully intact and the Drag & Drop / Prediction logic was 100% unaffected.
