# 5–7 Minute Capstone Demonstration Script

Use this as a speaking outline. Record your own screen and voice after running the program.

## 0:00–0:45 — Problem and goal
Explain that image processing applies similar arithmetic to many pixels, making it suitable for GPU parallelism. State the goal: convert RGB to grayscale and detect edges with Sobel.

## 0:45–1:45 — Algorithm
Explain grayscale conversion:
`Gray = 0.299R + 0.587G + 0.114B`.
Explain Sobel: two 3x3 masks estimate horizontal (`Gx`) and vertical (`Gy`) intensity changes. The edge magnitude is approximated with `abs(Gx) + abs(Gy)`.

## 1:45–3:00 — CUDA implementation
Show `GrayscaleKernel` and explain one thread per pixel. Show `SobelKernel` and explain 2D thread indexing, image-boundary checks, and neighboring pixel reads. Explain that image buffers are allocated in device global memory with `cudaMalloc`.

## 3:00–4:15 — Execution
Show the build command and actual terminal run. Run at least two image sizes and, if possible, multiple block sizes. Do not use invented output.

## 4:15–5:30 — Results and validation
Open the generated grayscale and Sobel images. Show the CSV timing file. Explain the actual mismatch count and GPU kernel time from your run. If the mismatch count is nonzero, investigate it before presenting.

## 5:30–6:30 — Lessons and next steps
Discuss the impact of image size and block dimensions. Explain that timing measures the kernels only, not image loading or memory transfers. A possible next optimization is shared-memory tiling to reuse neighboring pixels and reduce redundant global-memory reads.

## Recording checklist
- 5–10 minutes long.
- Show the project repository and README.
- Show a real build and run in a CUDA environment.
- Show generated output images and timing CSV.
- Explain the kernels in your own words.
- Upload the recording to a service that provides a reviewer-accessible URL.
