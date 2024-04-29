#include <dpct/dnnl_utils.hpp>
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <iostream>
#include "create_layer.h"

/**
 * Minimal example to apply sigmoid activation on a tensor 
 * using cuDNN.
 **/
int main(int argc, char** argv)
{    
    int numGPUs;
    numGPUs = dpct::dev_mgr::instance().device_count();
    std::cout << "Found " << numGPUs << " GPUs." << std::endl;
    /*
    DPCT1093:0: The "0" device may be not the one intended for use. Adjust the
    selected device if needed.
    */
    dpct::select_device(0); // use GPU0
    int device;
    dpct::device_info devProp;
    device = dpct::dev_mgr::instance().current_device_id();
    dpct::get_device_info(devProp,
                          dpct::dev_mgr::instance().get_device(device));
    /*
    DPCT1005:5: The SYCL device version is different from CUDA Compute
    Compatibility. You may need to rewrite this code.
    */
    std::cout << "Compute capability:" << devProp.get_major_version() << "."
              << devProp.get_minor_version() << std::endl;

    dpct::dnnl::engine_ext handle_;
    handle_.create_engine();
    std::cout << "Created cuDNN handle" << std::endl;

    // create the tensor descriptor
    dpct::library_data_t dtype = dpct::library_data_t::real_float;
    dpct::dnnl::memory_format_tag format = dpct::dnnl::memory_format_tag::nchw;
    int n = 1, c = 1, h = 1, w = 10;
    int NUM_ELEMENTS = n*c*h*w;

    // NOTE: do something with this !!!
    // dpct::dnnl::memory_desc_ext x_desc;
    l.x_desc = (dpct::dnnl::memory_desc_ext*)malloc(sizeof(dpct::dnnl::memory_desc_ext));
    // cast the pointer from void* to memory_desc_ext*
    dpct::dnnl::memory_desc_ext *x_desc = (dpct::dnnl::memory_desc_ext*)l.x_desc;


    /*
    DPCT1026:1: The call to cudnnCreateTensorDescriptor was removed because this
    functionality is redundant in SYCL.
    */
    (*x_desc).set(format, dtype, n, c, h, w);

    // create the tensor
    float *x;
    x = sycl::malloc_shared<float>(NUM_ELEMENTS, dpct::get_in_order_queue());
    for(int i=0;i<NUM_ELEMENTS;i++) x[i] = i * 1.00f;
    std::cout << "Original array: "; 
    for(int i=0;i<NUM_ELEMENTS;i++) std::cout << x[i] << " ";

    // create activation function descriptor
    float alpha[1] = {1};
    float beta[1] = {0.0};
    dpct::dnnl::activation_desc sigmoid_activation;
    dnnl::algorithm mode = dnnl::algorithm::eltwise_logistic_use_dst_for_bwd;
    /*
    DPCT1082:6: Migration of cudnnNanPropagation_t type is not supported.
    */
    /*
    DPCT1007:7: Migration of CUDNN_NOT_PROPAGATE_NAN is not supported.
    */
    // cudnnNanPropagation_t prop = CUDNN_NOT_PROPAGATE_NAN;
    /*
    DPCT1026:2: The call to cudnnCreateActivationDescriptor was removed because
    this functionality is redundant in SYCL.
    */
    /*
    DPCT1007:3: Migration of Nan numbers propagation option is not supported.
    */
    sigmoid_activation.set(mode, 0.0f);

    handle_.async_activation_forward(sigmoid_activation, *alpha, (*x_desc), x,
                                     *beta, *x_desc, x);

    dpct::get_in_order_queue().wait();
    // dpct::get_default_queue().wait();
    /*
    DPCT1026:4: The call to cudnnDestroy was removed because this functionality
    is redundant in SYCL.
    */
    std::cout << std::endl << "Destroyed cuDNN handle." << std::endl;
    std::cout << "New array: ";
    for(int i=0;i<NUM_ELEMENTS;i++) std::cout << x[i] << " ";
    std::cout << std::endl;
    dpct::dpct_free(x, dpct::get_in_order_queue());
    return 0;
}