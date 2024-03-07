CXX = /usr/local/cuda/bin/nvcc 
CXXFLAGS = -O3 -I/home/mazhaojia/pkg/cuda-samples/Common

SRC = graph.cu
TARGET = graph

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)

profile: $(TARGET)
	ncu $(TARGET)
