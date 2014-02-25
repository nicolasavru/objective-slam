#ifndef __PPF_UTILS_H
#define __PPF_UTILS_H

#include <cuda.h>
#include <cuda_runtime.h>                // Stops underlining of __global__
#include <device_launch_parameters.h>    // Stops underlining of threadIdx etc.
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/constant_iterator.h>

#include "debug.h"

template <typename Vector1, typename Vector2, typename Vector3>
void histogram(const Vector1& input,
               Vector2& histogram_values,
               Vector3& histogram_counts){
    typedef typename Vector1::value_type ValueType; // input value type
    typedef typename Vector3::value_type IndexType; // histogram index type

    thrust::device_vector<ValueType> data(input);
    thrust::sort(data.begin(), data.end());
    IndexType num_bins = thrust::inner_product(data.begin(), data.end() - 1,
                                               data.begin() + 1,
                                               IndexType(1),
                                               thrust::plus<IndexType>(),
                                               thrust::not_equal_to<ValueType>());
    histogram_values.resize(num_bins);
    histogram_counts.resize(num_bins);

    #ifdef DEBUG
        fprintf(stderr, "num_bins: %d\n", num_bins);
    #endif

    thrust::reduce_by_key(data.begin(), data.end(),
                          thrust::constant_iterator<IndexType>(1),
                          histogram_values.begin(),
                          histogram_counts.begin());
}

template <typename Vector1, typename Vector2, typename Vector3>
void histogram_destructive(Vector1& data,
                           Vector2& histogram_values,
                           Vector3& histogram_counts){
    typedef typename Vector1::value_type ValueType; // input value type
    typedef typename Vector3::value_type IndexType; // histogram index type

    thrust::sort(data.begin(), data.end());
    IndexType num_bins = thrust::inner_product(data.begin(), data.end() - 1,
                                               data.begin() + 1,
                                               IndexType(1),
                                               thrust::plus<IndexType>(),
                                               thrust::not_equal_to<ValueType>());

    #ifdef DEBUG
        fprintf(stderr, "num_bins: %d\n", num_bins);
    #endif

    histogram_values.resize(num_bins);
    histogram_counts.resize(num_bins);
    thrust::reduce_by_key(data.begin(), data.end(),
                          thrust::constant_iterator<IndexType>(1),
                          histogram_values.begin(),
                          histogram_counts.begin());
}

#endif /* __PPF_UTILS_H */
