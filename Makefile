NVCC ?= nvcc
NVCCFLAGS ?= -O2 -std=c++17 -lineinfo
TARGET = cuda_image_capstone

.PHONY: all clean run sample

all: $(TARGET)

$(TARGET): main.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

sample: $(TARGET)
	python3 generate_input.py --width 1024 --height 768 --output input.ppm
	./$(TARGET) --input input.ppm --output result --block 16

run: sample

clean:
	rm -f $(TARGET) input.ppm result_*.ppm result_*.csv
