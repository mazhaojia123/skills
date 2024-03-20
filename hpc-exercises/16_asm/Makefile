CXX = /usr/bin/g++ 
# INC = -I./
CXXFLAGS = -std=c++17 -O3 -save-temps -D__AVX512F__ -march=native
LDFLAG = -lm

SRC = hello_asm.cpp
EXEC = hello_asm
BUILD_PATH = ./build/
EXECS = $(addprefix $(BUILD_PATH), $(EXEC))

$(warning $(EXECS))

all: $(EXECS)

$(EXECS): $(SRC)
	mkdir -p build
	$(CXX) $(CXXFLAGS) $(LDFLAG) -o $@ $^

clean:
	rm -r ./build

run: 
	./build/$(TARGET)