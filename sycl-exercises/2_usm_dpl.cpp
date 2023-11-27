#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
using namespace sycl;
using namespace oneapi::dpl::execution;
const int N = 4;

int main() {
  queue q;
  std::cout << "Device : " << q.get_device().get_info<info::device::name>() << "\n";
    
  //# USM allocation on device
  int* data = malloc_device<int>(N, q);
  int* h_data = static_cast<int*>(malloc(sizeof(int)*N));
    
  //# Parallel STL algorithm using USM pointer
  oneapi::dpl::fill(make_device_policy(q), data, data + N, 20);
  q.wait();
  q.memcpy(h_data, data, N*sizeof(int));
  q.wait();
    
  for (int i = 0; i < N; i++) std::cout << h_data[i] << "\n";
  free(data, q);
  free(h_data);
  return 0;
}