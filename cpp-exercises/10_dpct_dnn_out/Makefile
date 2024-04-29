example: example.dp.cpp
	clang++ -fsycl -fsycl-targets=nvptx64-nvidia-cuda -o $@ -ldnnl $< 

clean: 
	rm example