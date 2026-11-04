"""Independent Python float64 transcription of pinned HF Qwen3.5 full attention.

Reference f2754311602414ed561cdf8ed9d852390f514404 modeling_qwen3_5_moe.py:
rotate_half/apply_rotary_pos_emb 648-688; repeat_kv/eager_attention_forward
691-731; Qwen3_5MoeAttention 734-815; RMSNorm 907-923; residual 972.
No production Rust imports. Synthetic F32 weights, no FTW packing claim.
Frozen pre-execution tolerance abs=2e-6 + rel=2e-5 for F32 reductions/exp.
Explicit output directory; at most 64 KiB generated artifacts.
"""
import hashlib
import json
import math
import pathlib
import struct
import sys

out = pathlib.Path(sys.argv[1])
if not out.is_dir():
    raise SystemExit("Supply existing caller-selected output directory")
f32 = lambda x: struct.unpack('<f', struct.pack('<f', x))[0]
def series(n, offset, scale):
    return [f32((((i * 7 + offset) % 29) - 14) * scale) for i in range(n)]
def matvec(w, x):
    return [sum(a*b for a,b in zip(w[i:i+len(x)], x)) for i in range(0,len(w),len(x))]
def norm(x, weight):
    inv = 1 / math.sqrt(sum(v*v for v in x) / len(x) + 1e-6)
    return [v * inv * (1+w) for v,w in zip(x,weight)]
def rope(x, position, adjacent=False):
    y = x[:]
    for i in range(2):
        a,b = (2*i,2*i+1) if adjacent else (i,i+2)
        angle = position * 10000 ** (-2*i/4)
        y[a] = x[a]*math.cos(angle)-x[b]*math.sin(angle)
        y[b] = x[b]*math.cos(angle)+x[a]*math.sin(angle)
    return y
inputs = {
    'q_weight': series(24*3,3,.025), 'k_weight': series(6*3,5,.035),
    'v_weight': series(6*3,9,.045), 'out_weight': series(3*12,11,.03),
    'q_norm': series(6,2,.02), 'k_norm': series(6,7,.025),
    'input_norm': series(3,8,.02),
    'hidden': [series(3,offset,.11) for offset in [1,6,12]],
}
keys,values,records=[],[],[]
for position,hidden in enumerate(inputs['hidden']):
    normalized = norm(hidden, inputs['input_norm'])
    projected = matvec(inputs['q_weight'], normalized)
    queries = [norm(projected[h*12:h*12+6],inputs['q_norm']) for h in range(2)]
    gates = [projected[h*12+6:h*12+12] for h in range(2)]
    key = norm(matvec(inputs['k_weight'],normalized),inputs['k_norm'])
    wrong_key = rope(key,position,True)
    keys.append(rope(key,position))
    values.append(matvec(inputs['v_weight'],normalized))
    rotated = [rope(q,position) for q in queries]
    probs,attended=[],[]
    for q in rotated:
        scores=[sum(a*b for a,b in zip(q,k))/math.sqrt(6) for k in keys]
        exps=[math.exp(x-max(scores)) for x in scores]
        p=[x/sum(exps) for x in exps]
        probs.extend(p)
        attended.extend(sum(p[t]*values[t][i] for t in range(len(p))) for i in range(6))
    flatgate=[x for head in gates for x in head]
    gated=[v/(1+math.exp(-g)) for v,g in zip(attended,flatgate)]
    output=matvec(inputs['out_weight'],gated)
    records.append(dict(normalized=normalized,projected=projected,
                        queries=[x for q in rotated for x in q],key=keys[-1],
                        wrong_adjacent_key=wrong_key,value=values[-1],probs=probs,
                        attended=attended,gated=gated,output=output,
                        residual=[a+b for a,b in zip(hidden,output)]))
assert max(abs(a-b) for a,b in zip(records[1]['key'],records[1]['wrong_adjacent_key'])) > .1
data=dict(inputs=inputs,records=records)
rust=['// Generated independently; see generate_qwen_attention_reference.py.']
def emit(name, values):
    rust.append('pub const '+name.upper()+': [f32; '+str(len(values))+'] = ['+', '.join(format(x,'.17g')+('_f32' if math.isfinite(x) else '') for x in values)+'];')
for name,value in inputs.items():
    if name=='hidden':
        for i,v in enumerate(value): emit('hidden_'+str(i),v)
    else: emit(name,value)
for i,record in enumerate(records):
    for name,value in record.items(): emit(name+'_'+str(i),value)
artifacts={'qwen_attention_fixture.rs':'\n'.join(rust)+'\n',
           'qwen_attention_reference.json':json.dumps(data,indent=2)+'\n'}
assert sum(len(x.encode()) for x in artifacts.values()) <= 65536
for name,content in artifacts.items():
    (out/name).write_bytes(content.encode('utf8'))
print(json.dumps({name:hashlib.sha256(content.encode()).hexdigest() for name,content in artifacts.items()}))
