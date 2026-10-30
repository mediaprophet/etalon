struct VertexOutput {
    float4 color : LOC5;
    float4 position : SV_Position;
    nointerpolation uint index : LOC0;
};

struct FragmentInput {
    nointerpolation uint index : LOC0;
    uint view : SV_ViewID;
    float4 color : LOC5;
    float4 position : SV_Position;
    bool front : SV_IsFrontFace;
};

struct VertexOutput_vertex_main {
    nointerpolation uint gen_naga_view_index : NAGA_VIEW_INDEX;
    nointerpolation uint index : LOC0;
    float4 color : LOC5;
    float4 position : SV_Position;
};

struct FragmentInput_fragment_main {
    nointerpolation uint view : NAGA_VIEW_INDEX;
    nointerpolation uint index_1 : LOC0;
    float4 color_1 : LOC5;
    float4 position_1 : SV_Position;
    bool front : SV_IsFrontFace;
};

VertexOutput ConstructVertexOutput(float4 arg0, float4 arg1, uint arg2) {
    VertexOutput ret = (VertexOutput)0;
    ret.color = arg0;
    ret.position = arg1;
    ret.index = arg2;
    return ret;
}

VertexOutput_vertex_main vertex_main(uint vertex : SV_VertexID, uint gen_naga_view_index : SV_ViewID)
{
    const VertexOutput vertexoutput = ConstructVertexOutput((0.5).xxxx, float4(0.0, 0.0, 0.0, 1.0), vertex);
    const VertexOutput_vertex_main vertexoutput_1 = { gen_naga_view_index, vertexoutput.index, vertexoutput.color, vertexoutput.position };
    return vertexoutput_1;
}

float4 fragment_main(FragmentInput_fragment_main fragmentinput_fragment_main) : SV_Target0
{
    FragmentInput input = { fragmentinput_fragment_main.index_1, fragmentinput_fragment_main.view, fragmentinput_fragment_main.color_1, fragmentinput_fragment_main.position_1, fragmentinput_fragment_main.front };
    return ((input.color * float((input.index + input.view))) + (input.position * float(input.front)));
}
