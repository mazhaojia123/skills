#include <sycl/sycl.hpp>
using namespace sycl;

static constexpr size_t N = 64; // global size : 一共多少线程
static constexpr size_t B = 32; // work-group size : 一个workgroup多少线程

int main() {
  queue q(gpu_selector_v);
  std::cout << "Device : " << q.get_device().get_info<info::device::name>() << "\n";

  q.submit([&](handler &h) {
    //# setup sycl stream class to print standard output from device code
    auto out = stream(10240, 7680, h);

    //# nd-range kernel
    h.parallel_for(nd_range<1>(N, B), [=](nd_item<1> item) {
      //# get sub_group handle
      auto wg = item.get_group();
      auto sg = item.get_sub_group();

      // # query sub_group and print sub_group info once per sub_group
      // if (sg.get_local_id()[0] == 0) {
        // out << "sub_group id: " << sg.get_group_id()[0] << " of "
        //     << sg.get_group_range()[0] << ", size=" << sg.get_local_range()[0]
        //     << "\n";
        out << "get_local_id: " << sg.get_local_id()
            << "; get_local_range: " << sg.get_local_range()
            << "; get_group_id: " << sg.get_group_id()
            << "; get_group_range: " << sg.get_group_range()
            << "; wg.get_group_id" << wg.get_group_id()
            << "; wg.get_group_range" << wg.get_group_range()
            << "; item.get_global_id" << item.get_global_id()
            << "\n";
      // }
    });
  }).wait();
}