#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void Get_Data_Statistics(
	__global double* data,

	__local double* local_sum,
	__local double* local_min,
	__local double* local_max,

	__global double* out_sum,
	__global double* out_min,
	__global double* out_max
)
{
	uint globalId = get_global_id(0);
	uint localSize = get_local_size(0);
	uint localId = get_local_id(0);
	uint groupId = get_group_id(0);

	local_sum[localId] = data[globalId];
	local_min[localId] = data[globalId];
	local_max[localId] = data[globalId];

	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = localSize >> 1; i > 0; i >>= 1)
	{
		if (localId < i)
		{
			local_sum[localId] += local_sum[localId + i];
			local_min[localId] = min(local_min[localId], local_min[localId + i]);
			local_max[localId] = max(local_max[localId], local_max[localId + i]);
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// Save/Agregate values
	if (localId == 0)
	{
		out_sum[groupId] = out_sum[groupId] + local_sum[0];
		out_min[groupId] = min(out_min[groupId], local_min[0]);
		out_max[groupId] = max(out_max[groupId], local_max[0]);
	}
}