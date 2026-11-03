#version 460
#extension GL_EXT_nonuniform_qualifier : require

struct _4
{
    uint _m0;
};

struct _5
{
    uint _m0;
};

struct _12
{
    uint _m0;
};

layout(set = 0, binding = 0, std430) readonly buffer _8_13
{
    uint _m0;
    _5 _m1;
    int _m2[];
} _13[10];

layout(set = 0, binding = 1, std430) readonly buffer _10_17
{
    uint _m0[];
} _17;

layout(set = 0, binding = 10, std140) uniform _20_19
{
    _4 _m0;
} _19;

layout(location = 0) flat in uint _24;
layout(location = 0) out uint _27;

void main()
{
    uint _35 = 0u;
    uint _41 = _12(_24)._m0;
    _35 += _13[0u]._m0;
    uint _56;
    if (_19._m0._m0 < 10u)
    {
        _56 = _13[_19._m0._m0]._m0;
    }
    else
    {
        _56 = 0u;
    }
    _35 += _56;
    uint _64;
    if (_41 < 10u)
    {
        _64 = _13[nonuniformEXT(_41)]._m0;
    }
    else
    {
        _64 = 0u;
    }
    _35 += _64;
    _35 += _13[7u]._m0;
    _35 += _13[0u]._m1._m0;
    uint _83;
    if (_19._m0._m0 < 10u)
    {
        _83 = _13[_19._m0._m0]._m1._m0;
    }
    else
    {
        _83 = 0u;
    }
    _35 += _83;
    uint _91;
    if (_41 < 10u)
    {
        _91 = _13[nonuniformEXT(_41)]._m1._m0;
    }
    else
    {
        _91 = 0u;
    }
    _35 += _91;
    _35 += _13[7u]._m1._m0;
    _35 += uint(_13[0u]._m2.length());
    uint _108;
    if (_19._m0._m0 < 10u)
    {
        _108 = uint(_13[_19._m0._m0]._m2.length());
    }
    else
    {
        _108 = 0u;
    }
    _35 += _108;
    uint _116;
    if (_41 < 10u)
    {
        _116 = uint(_13[nonuniformEXT(_41)]._m2.length());
    }
    else
    {
        _116 = 0u;
    }
    _35 += _116;
    _35 += uint(_13[7u]._m2.length());
    int _133;
    if (0u < uint(_13[0u]._m2.length()))
    {
        _133 = _13[0u]._m2[0u];
    }
    else
    {
        _133 = 0;
    }
    _35 += uint(_133);
    uint _142;
    if (_19._m0._m0 < 10u)
    {
        _142 = uint(_13[_19._m0._m0]._m2.length());
    }
    else
    {
        _142 = 0u;
    }
    int _150;
    if ((0u < _142) && (_19._m0._m0 < 10u))
    {
        _150 = _13[_19._m0._m0]._m2[0u];
    }
    else
    {
        _150 = 0;
    }
    _35 += uint(_150);
    uint _159;
    if (_41 < 10u)
    {
        _159 = uint(_13[nonuniformEXT(_41)]._m2.length());
    }
    else
    {
        _159 = 0u;
    }
    int _167;
    if ((0u < _159) && (_41 < 10u))
    {
        _167 = _13[nonuniformEXT(_41)]._m2[0u];
    }
    else
    {
        _167 = 0;
    }
    _35 += uint(_167);
    int _178;
    if (0u < uint(_13[7u]._m2.length()))
    {
        _178 = _13[7u]._m2[0u];
    }
    else
    {
        _178 = 0;
    }
    _35 += uint(_178);
    uint _189;
    if (0u < uint(_17._m0.length()))
    {
        _189 = _17._m0[0u];
    }
    else
    {
        _189 = 0u;
    }
    _35 += _189;
    _35 += uint(_17._m0.length());
    _27 = _35;
}

