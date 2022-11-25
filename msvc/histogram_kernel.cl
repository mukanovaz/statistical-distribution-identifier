#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

__kernel void Get_Data_Histogram(
    __global double* data,
    __local int* local_sums,
    __local double* local_var,

    __global double* out_sum,
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

    local_var[localId] = data[globalId];
    
    if (localId < bin_count) 
    {
        local_sums[localId] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //  Compute the interval number 
    if (globalId < bin_count)
    {
        int position = (int) (data[globalId] - min) * scale_factor;

        if (position == bin_count) 
        {
            position -= 1;
        }
        // increase the local count for interval 'position'
        atomic_inc(&local_sums[position]);
    }

    // Compute part of gauss variance
    for (int i = localSize >> 1; i > 0; i >>= 1)
    {
        if (localId < i)
        {
            double tmp = mean * data[globalId];
            local_var[localId] = tmp * tmp;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    //barrier(CLK_LOCAL_MEM_FENCE);

    if (localId < bin_count) {
        atomic_add(&out_sum[localId], local_sums[localId]);
    }

    if (localId == 0)
    {
        out_var[groupId] = local_var[0];
    }
}