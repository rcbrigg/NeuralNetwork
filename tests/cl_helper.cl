//#define TYPE float
//
//__kernel void upload(__global const float* src, __global TYPE* dst)
//{
//	const size_t gid = get_global_id(0);
//	vstore_half(src[gid], gid, dst);
//}
//
//__kernel void download(__global const half* src, __global TYPE* dst)
//{
//	const size_t gid = get_global_id(0);
//	dst[gid] = vload_half(gid, src);
//}