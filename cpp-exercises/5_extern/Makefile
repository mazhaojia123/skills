# 这一行不能用 % 
all: build_path build/main

build/main: build/main.o build/hello.o
	clang++ -o $@ $^

build/%.o : %.cpp
	clang++ -c -o $@ $^

run: 
	./build/main

clean: 
	rm -rf build

build_path: 
	mkdir -p build