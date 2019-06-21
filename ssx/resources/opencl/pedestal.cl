// Memset kernel to empty the counter array
kernel void memset_uint32(global unsigned int *ary, const int ngains, const int height, const int width)
{
    uint col = get_global_id(0);
    uint line = get_global_id(1);
    if ((col<width) && (line<height))
    {
        int offset = line*width+col;
        for (int pos=offset; pos<ngains*height*width; pos+=height*width)
        {
            ary[pos] = 0;
        }
    }
}

// Memset kernel to empty the sum & M2 arrays
kernel void memset_float32(global float *ary, const int ngains, const int height, const int width)
{
    uint col = get_global_id(0);
    uint line = get_global_id(1);
    if ((col<width) && (line<height))
    {
        int offset = line*width+col;
        for (int pos=offset; pos<ngains*height*width; pos+=height*width)
        {
            ary[pos] = 0.0f;
        }
    }
}


// allows the computation of the pedestal value from a set of frames.
kernel void feed(global ushort* raw,
                 const int bits,
                 const int ngains,
                 const int height,
                 const int width,
                 global float *sum_,
                 global float *M2,
                 global unsigned int *count
                )
{
    uint col = get_global_id(0);
    uint line = get_global_id(1);
    if ((col<width) && (line<height))
    {
        uint pos = col + width*line;
        ushort value = raw[pos];
        ushort gain = value>>bits;
        if (gain==3) gain=2;
        ushort mask = ((1<<bits) - 1);
        ushort trimmed = value & mask;
        float to_store = (float) trimmed;
        pos += gain * width * height;
        uint cnt = count[pos] + 1;
        count[pos] = cnt;
        float sm = sum_[pos] + to_store;
        sum_[pos] = sm;
        float delta = (sm/cnt)-to_store;
        M2[pos] += delta*delta*(cnt-1)/cnt;
    }


}

// Pedestal&gain correction
kernel void ocl_pedestal(global ushort *raw,
                         global float *gain,
                         global float *pedestal,
                         global float *result,
                         uint size)
{
    uint idx = get_global_id(0);
    if (idx<size)
    {
        ushort value = raw[idx];
        ushort g = value >>14;
        value &= (1<<14)-1;
        g = (g==3)?2:g;
        uint read_at = g*size + idx;
        float gain_value = gain[read_at];
        result[idx] = gain_value?(value - pedestal[read_at]) / gain_value:0.0f;
    }
}
