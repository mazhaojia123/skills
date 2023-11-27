#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/iterator>

using namespace sycl;
using namespace oneapi::dpl::execution;

static const int N = 4;


int main(){
  queue q(gpu_selector_v);
  std::cout << "Device : " << q.get_device().get_info<info::device::name>() << "\n";
//   std::vector<int> v{2,3,1,4};
  int v[4] = {2,3,1,4};
  int *d_v = malloc_device<int>(N,q);
  q.memcpy(d_v,v,N*sizeof(int));
  q.wait();
    
  //# Create a buffer and use buffer iterators in Parallel STL algorithms
  {

    oneapi::dpl::for_each(make_device_policy(q), d_v, d_v+N, [](int &a){ a *= 3; });
    oneapi::dpl::sort(make_device_policy(q), d_v, d_v+N);
  }
  q.wait();
  q.memcpy(v,d_v,N*sizeof(int));
  q.wait();
    
  for(int i = 0; i < N; i++) std::cout << v[i] << "\n";
  return 0;
}