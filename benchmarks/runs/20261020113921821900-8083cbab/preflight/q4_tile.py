"""Exact half2 Q4 dequantization prototype, to be owned by the weight codec."""
CUDA_SOURCE = r'''
__device__ __forceinline__ unsigned int akas_q4_half2(unsigned int packed, unsigned int scale) {
    unsigned int values = (packed & 15u) | ((packed >> 4u) << 16u) | 0x64006400u;
    unsigned int integers, result;
    asm("sub.f16x2 %0, %1, %2;" : "=r"(integers) : "r"(values), "r"(0x64086408u));
    asm("mul.f16x2 %0, %1, %2;" : "=r"(result) : "r"(integers), "r"(scale | (scale << 16u)));
    return result;
}
'''

def tile_reader(depth):
    import tilelang.language as T
    @T.macro
    def load(encoded, destination, row, column):
        T.import_source(CUDA_SOURCE)
        for i,j in T.Parallel(destination.shape[0],destination.shape[1]//2):
            channel=column+j//16*32+j%16
            base=((row+i)*depth+channel)//32*18
            scale=T.cast(encoded[0][base],'uint32') | T.cast(encoded[0][base+1],'uint32') << 8
            packed=T.cast(encoded[0][base+2+channel%16],'uint32')
            pair=T.call_extern('uint32','akas_q4_half2',packed,scale)
            destination[i,j//16*32+j%16]=T.reinterpret('float16',T.cast(pair,'uint16'))
            destination[i,j//16*32+j%16+16]=T.reinterpret('float16',T.cast(pair>>16,'uint16'))
    return load
