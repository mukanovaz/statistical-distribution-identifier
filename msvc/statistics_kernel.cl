#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void Get_Data_Statistics(
	__global double* data, 
	__local double* local_new_m,
	__local double* local_new_s,
	__local double* local_n,
	__local double* local_sum,
	__local double* local_sum_abs,
	__local double* local_min,
	__local double* local_max,
	__local double* local_data

	__global double* out_new_m,
	__global double* out_new_s,
	__global double* out_n,
	__global double* out_sum,
	__global double* out_sum_abs,
	__global double* out_min,
	__global double* out_max,
	__global double* out_data )
{		
	size_t globalId = get_global_id(0);
	size_t localSize = get_local_size(0);
	size_t localId = get_local_id(0);

	local_sum[localId] = data[globalId];
	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = localSize >> 1; i > 0; i >>= 1)
	{
		if (localId < i)
		{
			local_sum[localId] += local_sum[localId + 1];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (localId == 0)
	{
		out_sum[get_group_id(0)] = localData[0];
	}
}