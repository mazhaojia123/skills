NVCC = nvcc
NVCCFLAGS = -O3 -arch=sm_80 -std=c++11
LIBS = -lcublas

TARGET = gemv
SOURCES = gemv.cu

all: $(TARGET)

$(TARGET): $(SOURCES) kernel0.cuh
	$(NVCC) $(NVCCFLAGS) $(SOURCES) -o $(TARGET) $(LIBS)

clean:
	rm -f $(TARGET)

test: $(TARGET)
	./$(TARGET)

.PHONY: all clean test
