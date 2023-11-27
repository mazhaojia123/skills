#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/iterator>
#include <chrono>

using namespace sycl;
using namespace oneapi::dpl::execution;

int main(){
  queue q(cpu_selector_v);
  std::cout << "Device : " << q.get_device().get_info<info::device::name>() << "\n";
  std::vector<int> v{2,3,1,4};

  // Start the timer
  auto start = std::chrono::high_resolution_clock::now();

  // Launch the kernel again
  //# Create a buffer and use buffer iterators in Parallel STL algorithms
  {
    buffer buf(v);
    auto buf_begin = oneapi::dpl::begin(buf);
    auto buf_end   = oneapi::dpl::end(buf);

    oneapi::dpl::for_each(make_device_policy(q), buf_begin, buf_end, [](int &a){ a *= 3; });
    oneapi::dpl::sort(make_device_policy(q), buf_begin, buf_end);

    q.wait();
  }

  // Stop the timer
  auto stop = std::chrono::high_resolution_clock::now();

  // Calculate the elapsed time in milliseconds
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
  std::cout << "Elapsed time: " << duration.count() << " ms" << std::endl;
    
    
  for(int i = 0; i < v.size(); i++) std::cout << v[i] << "\n";
  return 0;
}