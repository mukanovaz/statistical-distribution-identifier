#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void Get_Data_Statistics(
	__global double* data, 

	__local double* local_sum,
	__local double* local_sumAbs,
	__local double* local_min,
	__local double* local_max,

	__global double* out_sum,
	__global double* out_sumAbs,
	__global double* out_min,
	__global double* out_max
	)
{		
	size_t globalId = get_global_id(0);
	size_t localSize = get_local_size(0);
	size_t localId = get_local_id(0);

	local_sum[localId] = data[globalId];
	local_sumAbs[localId] = data[globalId];
	local_min[localId] = data[globalId];
	local_max[localId] = data[globalId];

	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = localSize >> 1; i > 0; i >>= 1)
	{
		if (localId < i)
		{
			local_sum[localId] += local_sum[localId + i];
			local_sumAbs[localId] += fabs(local_sumAbs[localId + i]);
			local_min[localId] = min(local_min[localId], local_min[localId + i]);
			local_max[localId] = max(local_max[localId], local_max[localId + i]);
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (localId == 0)
	{
		out_sum[get_group_id(0)] = local_sum[0];
		out_sumAbs[get_group_id(0)] = local_sumAbs[0];
		out_min[get_group_id(0)] = local_min[0];
		out_max[get_group_id(0)] = local_max[0];
	}
}