#ifndef __PPF_UTILS_H
#define __PPF_UTILS_H

#include <cuda.h>
#include <cuda_runtime.h>                // Stops underlining of __global__
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <iostream>

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

    // thrust::sort(data.begin(), data.end());
    IndexType num_bins = thrust::inner_product(data.begin(), data.end() - 1,
                                               data.begin() + 1,
                                               IndexType(1),
                                               thrust::plus<IndexType>(),
                                               thrust::not_equal_to<ValueType>());

    // #ifdef DEBUG
    // fprintf(stderr, "num_bins: %d\n", num_bins);
    // #endif

    histogram_values.resize(num_bins);
    histogram_counts.resize(num_bins);
    thrust::reduce_by_key(data.begin(), data.end(),
                          thrust::constant_iterator<IndexType>(1),
                          histogram_values.begin(),
                          histogram_counts.begin());
}

// http://stackoverflow.com/questions/17698969/cuda-thrust-find-index-of-minimum-value-among-multiple-vectors
template <typename Vector1, typename Vector2>
void rowwise_max(Vector1& input, int rows, int cols,
                 Vector2& maxidx){
    typedef typename Vector1::value_type ValueType; // input value type

    // allows us to use "_1" instead of "thrust::placeholders::_1"
    using namespace thrust::placeholders;

    thrust::device_vector<float> maxval(rows);

    // OH GOD HOW DO YOU FORMAT THIS?!?
    thrust::reduce_by_key
        (// key input: step function that increments for every row
         thrust::make_transform_iterator(thrust::make_counting_iterator(0), _1 / cols),
         thrust::make_transform_iterator(thrust::make_counting_iterator(0), _1 / cols) + rows*cols,
         // value input: (value, index) tuple
         thrust::make_zip_iterator
         (thrust::make_tuple(input.begin(),
                             thrust::make_transform_iterator(thrust::make_counting_iterator(0),
                                                             _1 % cols))),
         // discard key output
         thrust::make_discard_iterator(),
         thrust::make_zip_iterator(thrust::make_tuple(maxval.begin(),
                                                      maxidx.begin())),
         thrust::equal_to<int>(),
         // compare by first element of tuple
         thrust::maximum<thrust::tuple<ValueType, int> >()
    );

}
#endif /* __PPF_UTILS_H */
