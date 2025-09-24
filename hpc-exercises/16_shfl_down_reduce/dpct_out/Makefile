main : main.dp.cpp
	clang++ -o $@ $^ -fsycl -fsycl-targets=nvptx64-nvidia-cuda
clean: 
	rm main
run: main
	./main

