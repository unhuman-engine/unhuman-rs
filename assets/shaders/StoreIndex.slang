// The smallest compute shader that proves a dispatch ran: one invocation writes
// a known value, so the test reads it back instead of trusting that a dispatch
// with no visible effect succeeded.
//
// The buffer is read-write, not read-only. wgpu validates the layout against the
// shader's access, so a read-only binding would be refused at pipeline creation
// and the dispatch would never be the thing under test.

[[vk::binding(0, 0)]]
RWStructuredBuffer<uint> output;

[numthreads(1, 1, 1)]
[shader("compute")]
void cs_main(uint3 id : SV_DispatchThreadID)
{
    output[0] = 7u;
}
