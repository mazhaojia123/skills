#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdio.h>

void warpReduce(int *out, const sycl::nd_item<3> &item_ct1) {
        int lane_id = item_ct1.get_local_id(2) & 0x1f;
        int value = 1;
	
	for (int i = 16; i >= 1; i /= 2)
                value += dpct::shift_sub_group_left(item_ct1.get_sub_group(), value, i);

        out[lane_id]=value;
}

int main() {
 dpct::device_ext &dev_ct1 = dpct::get_current_device();
 sycl::queue &q_ct1 = dev_ct1.in_order_queue();
        int tmp_h[32];
	int *tmp_d;
        tmp_d = sycl::malloc_device<int>(32, q_ct1);
        q_ct1.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, 32),
                                             sycl::range<3>(1, 1, 32)),
                           [=](sycl::nd_item<3> item_ct1)
                               [[intel::reqd_sub_group_size(32)]] {
                                       warpReduce(tmp_d, item_ct1);
                               });
        q_ct1.memcpy(tmp_h, tmp_d, 32 * sizeof(int)).wait();
        dev_ct1.queues_wait_and_throw();
        printf("result: %d\n", tmp_h[0]);
	return 0;
}
