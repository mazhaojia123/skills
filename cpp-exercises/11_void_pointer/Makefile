example: create_layer.o example.o
	clang++ -fsycl -fsycl-targets=nvptx64-nvidia-cuda -o $@ -ldnnl $^

%.o: %.dp.cpp
	clang++ -c -fsycl -fsycl-targets=nvptx64-nvidia-cuda -o $@ $<

%.o: %.c
	clang -c -o $@ $<

clean: 
	rm example *.o