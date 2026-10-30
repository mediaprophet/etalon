float4 vertex_main() : SV_Position
{
    return float4(0.0, 0.0, 0.0, 1.0);
}

float4 fragment_main() : SV_Target0
{
    return (1.0).xxxx;
}
