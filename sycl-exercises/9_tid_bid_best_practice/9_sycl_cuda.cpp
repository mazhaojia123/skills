#include <sycl/sycl.hpp>
using namespace sycl;

static constexpr size_t N = 64;  // global size : 一共多少线程
static constexpr size_t B = 32;  // work-group size : 一个workgroup多少线程

int main() {
    queue q(gpu_selector_v);
    std::cout << "Device : " << q.get_device().get_info<info::device::name>()
              << "\n";

    q.submit([&](handler &h) {
		// # setup sycl stream class to print standard output from device code
		auto out = stream(10240, 7680, h);

		sycl::range<3> blockDim(1,2,3);
		sycl::range<3> threadDim(4,5,6);
		// # nd-range kernel
		h.parallel_for(
		nd_range<3>(blockDim * threadDim, threadDim),
		[=](sycl::nd_item<3> item) {
			out << "gridDim(2,1,0): "	 << item.get_group_range(2) << ',' << item.get_group_range(1) << ',' << item.get_group_range(0)
				<< "\tblockIdx(2,1,0): " << item.get_group(2) << ',' << item.get_group(1) << ',' << item.get_group(0) 
				// << "\tblockDim(2,1,0): " << item.get_local_range().get(2) << ',' << item.get_local_range().get(1) << ',' << item.get_local_range().get(0)
				<< "\tblockDim(2,1,0): " << item.get_local_range(2) << ',' << item.get_local_range(1) << ',' << item.get_local_range(0)
				<< "\tthreadIdx(2,1,0): "<< item.get_local_id(2) << ',' << item.get_local_id(1) << ',' << item.get_local_id(0)
				<< '\n';
		}
		);
	}).wait();
}