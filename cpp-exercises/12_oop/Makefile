CXX := g++
CXXFLAGS := -std=c++17 -Wall -Wextra -O2
TARGET := oop_demo
SRC := main.cpp

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $^ -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
