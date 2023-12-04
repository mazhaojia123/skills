CXX = /usr/local/cuda-11.7/bin/nvcc 
CXXFLAGS = -I./ -lcublas -arch=sm_70

SRC = main.cu
TARGET = main

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)

profile: $(TARGET)
	ncu $(TARGET)
