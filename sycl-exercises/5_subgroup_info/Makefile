CXX = /home/mazhaojia/pkg/sycl_workspace/build-20240301/bin/clang++ 
# LD = -L/home/mazhaojia/sycl_workspace/build-20230826-slg/lib 
# INC = -I./Common

# CXXFLAGS = -fsycl -fsycl-targets=nvptx64-nvidia-cuda -O3 $(LD) $(INC)
CXXFLAGS = -fsycl -fsycl-targets=nvptx64-nvidia-cuda
SRC = 5_subgroup_info.cpp
TARGET = 5_subgroup_info

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)
