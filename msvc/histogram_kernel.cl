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

    double val = data[globalId];

    val = min < 0 ? val / 2 : val;

    // Increase number on histogram position
    int position = (int)((val - min) * scale_factor);
    uint old_val = atomic_inc(&out_sum[2 * position]);
    atomic_add(&out_sum[2 * position + 1], old_val == 0xFFFFFFFF);

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