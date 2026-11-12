#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t error = (call);                                              \
    if (error != cudaSuccess) {                                              \
      throw std::runtime_error(std::string("CUDA error: ") +                 \
                               cudaGetErrorString(error));                   \
    }                                                                        \
  } while (0)

struct Image {
  int width = 0;
  int height = 0;
  std::vector<uint8_t> pixels;
};

__global__ void GrayscaleKernel(const uint8_t* rgb, uint8_t* gray,
                                int pixel_count) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= pixel_count) return;

  int offset = index * 3;
  float value = 0.299f * rgb[offset] + 0.587f * rgb[offset + 1] +
                0.114f * rgb[offset + 2];
  gray[index] = static_cast<uint8_t>(value);
}

__global__ void SobelKernel(const uint8_t* gray, uint8_t* edges,
                            int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;

  int index = y * width + x;
  if (x == 0 || y == 0 || x == width - 1 || y == height - 1) {
    edges[index] = 0;
    return;
  }

  int gx = -gray[(y - 1) * width + (x - 1)] -
           2 * gray[y * width + (x - 1)] -
           gray[(y + 1) * width + (x - 1)] +
           gray[(y - 1) * width + (x + 1)] +
           2 * gray[y * width + (x + 1)] +
           gray[(y + 1) * width + (x + 1)];

  int gy = -gray[(y - 1) * width + (x - 1)] -
           2 * gray[(y - 1) * width + x] -
           gray[(y - 1) * width + (x + 1)] +
           gray[(y + 1) * width + (x - 1)] +
           2 * gray[(y + 1) * width + x] +
           gray[(y + 1) * width + (x + 1)];

  int magnitude = abs(gx) + abs(gy);
  edges[index] = static_cast<uint8_t>(min(magnitude, 255));
}

Image ReadPpm(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("Cannot open input image: " + path);

  std::string magic;
  input >> magic;
  if (magic != "P6") {
    throw std::runtime_error("Input must be a binary PPM (P6) image.");
  }

  Image image;
  int max_value = 0;
  input >> image.width >> image.height >> max_value;
  input.get();

  if (image.width <= 0 || image.height <= 0 || max_value != 255) {
    throw std::runtime_error("Invalid PPM dimensions or max value.");
  }

  image.pixels.resize(static_cast<size_t>(image.width) * image.height * 3);
  input.read(reinterpret_cast<char*>(image.pixels.data()),
             static_cast<std::streamsize>(image.pixels.size()));
  if (!input) throw std::runtime_error("Could not read complete PPM image.");
  return image;
}

void WritePpm(const std::string& path, int width, int height,
              const std::vector<uint8_t>& pixels, int channels) {
  std::ofstream output(path, std::ios::binary);
  if (!output) throw std::runtime_error("Cannot write output image: " + path);
  output << "P6\n" << width << " " << height << "\n255\n";
  if (channels == 3) {
    output.write(reinterpret_cast<const char*>(pixels.data()),
                 static_cast<std::streamsize>(pixels.size()));
  } else {
    for (uint8_t value : pixels) {
      output.put(static_cast<char>(value));
      output.put(static_cast<char>(value));
      output.put(static_cast<char>(value));
    }
  }
}

std::vector<uint8_t> CpuGrayscale(const Image& image) {
  size_t count = static_cast<size_t>(image.width) * image.height;
  std::vector<uint8_t> gray(count);
  for (size_t i = 0; i < count; ++i) {
    size_t offset = i * 3;
    float value = 0.299f * image.pixels[offset] +
                  0.587f * image.pixels[offset + 1] +
                  0.114f * image.pixels[offset + 2];
    gray[i] = static_cast<uint8_t>(value);
  }
  return gray;
}

std::vector<uint8_t> CpuSobel(const std::vector<uint8_t>& gray,
                              int width, int height) {
  std::vector<uint8_t> edges(static_cast<size_t>(width) * height, 0);
  for (int y = 1; y < height - 1; ++y) {
    for (int x = 1; x < width - 1; ++x) {
      int gx = -gray[(y - 1) * width + x - 1] -
               2 * gray[y * width + x - 1] -
               gray[(y + 1) * width + x - 1] +
               gray[(y - 1) * width + x + 1] +
               2 * gray[y * width + x + 1] +
               gray[(y + 1) * width + x + 1];
      int gy = -gray[(y - 1) * width + x - 1] -
               2 * gray[(y - 1) * width + x] -
               gray[(y - 1) * width + x + 1] +
               gray[(y + 1) * width + x - 1] +
               2 * gray[(y + 1) * width + x] +
               gray[(y + 1) * width + x + 1];
      edges[y * width + x] =
          static_cast<uint8_t>(std::min(std::abs(gx) + std::abs(gy), 255));
    }
  }
  return edges;
}

int main(int argc, char* argv[]) {
  try {
    std::string input_path = "input.ppm";
    std::string output_prefix = "result";
    int block_size = 16;

    for (int i = 1; i < argc; ++i) {
      std::string arg = argv[i];
      if (i + 1 >= argc) throw std::runtime_error("Missing value for " + arg);
      std::string value = argv[++i];
      if (arg == "--input") input_path = value;
      else if (arg == "--output") output_prefix = value;
      else if (arg == "--block") block_size = std::stoi(value);
      else throw std::runtime_error("Unknown argument: " + arg);
    }

    if (block_size < 1 || block_size > 32) {
      throw std::runtime_error("--block must be between 1 and 32.");
    }

    Image image = ReadPpm(input_path);
    const int pixel_count = image.width * image.height;
    const size_t rgb_bytes = image.pixels.size();
    const size_t gray_bytes = static_cast<size_t>(pixel_count);

    uint8_t *d_rgb = nullptr, *d_gray = nullptr, *d_edges = nullptr;
    CUDA_CHECK(cudaMalloc(&d_rgb, rgb_bytes));
    CUDA_CHECK(cudaMalloc(&d_gray, gray_bytes));
    CUDA_CHECK(cudaMalloc(&d_edges, gray_bytes));
    CUDA_CHECK(cudaMemcpy(d_rgb, image.pixels.data(), rgb_bytes,
                          cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    int threads = 256;
    int blocks = (pixel_count + threads - 1) / threads;
    GrayscaleKernel<<<blocks, threads>>>(d_rgb, d_gray, pixel_count);
    CUDA_CHECK(cudaGetLastError());

    dim3 sobel_block(block_size, block_size);
    dim3 sobel_grid((image.width + block_size - 1) / block_size,
                    (image.height + block_size - 1) / block_size);
    SobelKernel<<<sobel_grid, sobel_block>>>(d_gray, d_edges,
                                             image.width, image.height);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float gpu_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, start, stop));

    std::vector<uint8_t> gray(gray_bytes), edges(gray_bytes);
    CUDA_CHECK(cudaMemcpy(gray.data(), d_gray, gray_bytes,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(edges.data(), d_edges, gray_bytes,
                          cudaMemcpyDeviceToHost));

    std::vector<uint8_t> cpu_gray = CpuGrayscale(image);
    std::vector<uint8_t> cpu_edges = CpuSobel(cpu_gray, image.width,
                                               image.height);
    size_t mismatch = 0;
    for (size_t i = 0; i < gray.size(); ++i) {
      if (gray[i] != cpu_gray[i]) ++mismatch;
      if (edges[i] != cpu_edges[i]) ++mismatch;
    }

    WritePpm(output_prefix + "_grayscale.ppm", image.width, image.height,
             gray, 1);
    WritePpm(output_prefix + "_sobel.ppm", image.width, image.height,
             edges, 1);

    std::ofstream csv(output_prefix + "_timings.csv");
    csv << "width,height,pixels,block_size,gpu_kernel_time_ms,validation_mismatches\n";
    csv << image.width << "," << image.height << "," << pixel_count << ","
        << block_size << "," << std::setprecision(8) << gpu_ms << ","
        << mismatch << "\n";

    std::cout << "Input: " << input_path << "\n";
    std::cout << "Image dimensions: " << image.width << " x " << image.height
              << "\n";
    std::cout << "Pixels processed: " << pixel_count << "\n";
    std::cout << "CUDA kernel time (ms): " << gpu_ms << "\n";
    std::cout << "CPU/GPU validation mismatches: " << mismatch << "\n";
    std::cout << "Wrote " << output_prefix << "_grayscale.ppm, "
              << output_prefix << "_sobel.ppm, and "
              << output_prefix << "_timings.csv\n";

    CUDA_CHECK(cudaFree(d_rgb));
    CUDA_CHECK(cudaFree(d_gray));
    CUDA_CHECK(cudaFree(d_edges));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return mismatch == 0 ? 0 : 2;
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << "\n";
    return 1;
  }
}
