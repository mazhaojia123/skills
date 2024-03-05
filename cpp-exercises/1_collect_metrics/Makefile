CXX = /usr/bin/g++ 
# INC = -I./
# CXXFLAGS = 

SRC = main.cpp func.cpp
TARGET = main

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(SRC) -o $(TARGET)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)