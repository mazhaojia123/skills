#include <sycl/sycl.hpp>

using namespace sycl;

int main() {
  queue q(gpu_selector_v);

  //# Print the device info
  std::cout << "device name   : " << q.get_device().get_info<info::device::name>() << "\n";
  std::cout << "local_mem_size: " << q.get_device().get_info<info::device::local_mem_size>() << "\n";

  auto local_mem_type = q.get_device().get_info<info::device::local_mem_type>();
  if(local_mem_type == info::local_mem_type::local) 		// 有单独的内存区域
    std::cout << "local_mem_type: info::local_mem_type::local" << "\n";
  else if(local_mem_type == info::local_mem_type::global) 	 // 没有在缓存上开辟单独区域
    std::cout << "local_mem_type: info::local_mem_type::global" << "\n";
  else if(local_mem_type == info::local_mem_type::none) 
    std::cout << "local_mem_type: info::local_mem_type::none" << "\n";
 
  return 0;
}