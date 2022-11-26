#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

__kernel void Get_Data_Histogram(
    __global double* data,
    __local double* local_var,

    __global int* out_sum,
    __global double* out_var,
    double mean,
    double min,
    double scale_factor,
    double bin_size,
    long   bin_count
)
{
    uint localId = get_local_id(0);
    uint globalId = get_global_id(0);
    uint groupSize = get_local_size(0);
    uint groupId = get_group_id(0);
    uint localSize = get_local_size(0);

    local_var[localId] = (data[globalId] - mean) * (data[globalId] - mean);
    
    barrier(CLK_LOCAL_MEM_FENCE);

    //  Compute the interval number 
    int position = (int)((data[globalId] - min) * scale_factor);

    if (position == bin_count)
    {
        position -= 1;
    }

    // increase the hist count for interval 'position'
    uint val = atomic_inc(&out_sum[position]);
    atomic_add(&out_sum[position], val == 0xFFFFFFFF);
  
    // Compute part of gauss variance
    for (int i = 1; i < groupSize; i++)
    {
        double tmp = (double)data[globalId + i] - (double)mean;
        local_var[localId] += tmp * tmp;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (localId == 0)
    {
        out_var[groupId] = local_var[0];
    }
}