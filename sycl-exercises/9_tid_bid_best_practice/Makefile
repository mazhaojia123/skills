DP_TARGET = 9_sycl_cuda

all: dpcpp

build:
	mkdir build

dpcpp: CXX = /home/mazhaojia/pkg/sycl_workspace/build-20240301/bin/clang++
dpcpp: CXXFLAGS = -fsycl -fsycl-targets=nvptx64-nvidia-cuda
dpcpp: build
	$(CXX) $(CXXFLAGS) -o build/$(DP_TARGET) $(DP_TARGET).cpp

clean:
	rm -r build

run: all
	./build/$(DP_TARGET)
