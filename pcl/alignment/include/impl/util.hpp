#ifndef UTIL_IMPL_H
#define UTIL_IMPL_H

#include <Eigen/Core>
#include <thrust/device_vector.h>

template <typename T>
void write_array(const char *filename, T *data, int n){
    FILE *fp;

    if(!(fp = fopen(filename,"wb"))){
        fprintf(stderr, "Error opening file file %s.\n", filename);
        exit(1);
    }

    if(fwrite(data, sizeof(T), n, fp) != n){
        fprintf(stderr, "Error writing to file %s.\n", filename);
        exit(2);
    }

    if(fclose(fp)){
        fprintf(stderr, "Error closing file %s.\n", filename);
        exit(3);
    }
}

template <typename T>
void write_device_array(const char *filename, T *data, int n){
    T *host_array = new T[n];
    if(cudaMemcpy(host_array, data, n*sizeof(T), cudaMemcpyDeviceToHost)
       != cudaSuccess){
        fprintf(stderr, "Error copying data from device to host.");
        exit(4);
    }
    write_array(filename, host_array, n);
    delete []host_array;
}

template <typename T>
void write_device_vector(const char *filename, thrust::device_vector<T> *data){
    write_device_array(filename, RAW_PTR(data), data->size());
}

// http://eigen.tuxfamily.org/bz/show_bug.cgi?id=622
template<typename Derived>
std::istream &operator>>(std::istream &s,
                           Eigen::MatrixBase<Derived> &m){
    for(int i = 0; i < m.rows(); ++i){
        for(int j = 0; j < m.cols(); j++){
            s >> m(i,j);
        }
    }
    return s;
}


#endif /* UTIL_IMPL_H */
