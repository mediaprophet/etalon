struct VertexInput {
    uint vertex : SV_VertexID;
    uint view : SV_ViewID;
};

struct VertexOutput {
    float4 position : SV_Position;
    float unused : LOC0;
};

struct FragmentInput {
    float4 position : SV_Position;
    uint view : SV_ViewID;
};

struct VertexOutput_scalar_output {
    nointerpolation uint gen_naga_view_index : NAGA_VIEW_INDEX;
    precise float4 position : SV_Position;
};

struct VertexOutput_scalar_existing_view {
    nointerpolation uint gen_naga_view_index_1 : NAGA_VIEW_INDEX;
    float4 position_1 : SV_Position;
};

struct VertexOutput_struct_existing_view {
    nointerpolation uint gen_naga_view_index_2 : NAGA_VIEW_INDEX;
    float4 position_2 : SV_Position;
};

struct FragmentInput_fragment_view {
    nointerpolation uint view_2 : NAGA_VIEW_INDEX;
};

struct FragmentInput_fragment_struct {
    nointerpolation uint view_3 : NAGA_VIEW_INDEX;
    float4 position_3 : SV_Position;
};

VertexOutput_scalar_output scalar_output(uint gen_naga_view_index : SV_ViewID)
{
    const VertexOutput_scalar_output gen_naga_output = { gen_naga_view_index, float4(0.0, 0.0, 0.0, 1.0) };
    return gen_naga_output;
}

VertexOutput_scalar_existing_view scalar_existing_view(uint view : SV_ViewID)
{
    uint gen_naga_view_index_1 = view;
    if ((view == 0u)) {
        const VertexOutput_scalar_existing_view gen_naga_output_1 = { gen_naga_view_index_1, float4(0.0, 0.0, 0.0, 1.0) };
        return gen_naga_output_1;
    }
    const VertexOutput_scalar_existing_view gen_naga_output_2 = { gen_naga_view_index_1, float4(1.0, 0.0, 0.0, 1.0) };
    return gen_naga_output_2;
}

VertexOutput ConstructVertexOutput(float4 arg0, float arg1) {
    VertexOutput ret = (VertexOutput)0;
    ret.position = arg0;
    ret.unused = arg1;
    return ret;
}

VertexOutput_struct_existing_view struct_existing_view(VertexInput input)
{
    uint gen_naga_view_index_2 = input.view;
    const VertexOutput vertexoutput = ConstructVertexOutput(float4(float(input.view), float(input.vertex), 0.0, 1.0), 3.0);
    const VertexOutput_struct_existing_view vertexoutput_1 = { gen_naga_view_index_2, vertexoutput.position };
    return vertexoutput_1;
}

float4 fragment_view(FragmentInput_fragment_view fragmentinput_fragment_view) : SV_Target0
{
    uint view_1 = fragmentinput_fragment_view.view_2;
    return (float(view_1)).xxxx;
}

float4 fragment_struct(FragmentInput_fragment_struct fragmentinput_fragment_struct) : SV_Target0
{
    FragmentInput input_1 = { fragmentinput_fragment_struct.position_3, fragmentinput_fragment_struct.view_3 };
    return ((float(input_1.view)).xxxx + input_1.position);
}

float4 fragment_without_view() : SV_Target0
{
    return (1.0).xxxx;
}
