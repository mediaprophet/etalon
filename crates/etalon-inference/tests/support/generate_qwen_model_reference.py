"""Independent mixed Qwen3.5 scalar float64 oracle; no production imports.
HF f2754311602414ed561cdf8ed9d852390f514404 modeling_qwen3_5_moe.py.
Frozen tolerance before Rust execution: 2e-6 + 2e-5*abs(expected).
Synthetic F32 fixture, not checkpoint packing/full-model qualification.
"""
import math, struct, pathlib, sys, json, hashlib
out=pathlib.Path(sys.argv[1])
assert out.is_dir()
f32=lambda x:struct.unpack('<f',struct.pack('<f',x))[0]
def series(n,offset,scale=.025): return [f32((((i*7+offset)%29)-14)*scale) for i in range(n)]
weights={}
def add(name,dims,offset,scale=.025):
    x=series(math.prod(dims),offset,scale);weights[name]=(dims,x);return x
def w(name):return weights[name][1]
def mv(a,x):return [sum(v*z for v,z in zip(a[i:i+len(x)],x)) for i in range(0,len(a),len(x))]
def silu(x):return x/(1+math.exp(-x))
def norm(x,a):return [v*(1+t)/math.sqrt(sum(z*z for z in x)/len(x)+1e-6) for v,t in zip(x,a)]
def l2(x):return [v/math.sqrt(sum(z*z for z in x)+1e-6) for v in x]
def softmax(x):
    e=[math.exp(v-max(x)) for v in x];return [v/sum(e) for v in e]
def rope(x,p):
    y=x[:]
    for i in range(2):
        a=p*[1,.01][i];y[i]=x[i]*math.cos(a)-x[i+2]*math.sin(a);y[i+2]=x[i+2]*math.cos(a)+x[i]*math.sin(a)
    return y
for l in range(2):
    prefix=f'model.layers.{l}.'
    add(prefix+'input_layernorm.weight',[3],3+l)
    add(prefix+'post_attention_layernorm.weight',[3],7+l)
    for name,dims,offset in [('gate',[3,2],1),('experts.gate_proj',[3,2,2],5),('experts.up_proj',[3,2,2],9),('experts.down_proj',[2,3,2],13),('shared_expert.gate_proj',[3,2],17),('shared_expert.up_proj',[3,2],20),('shared_expert.down_proj',[2,3],23),('shared_expert_gate',[3],26)]:
        add(prefix+'mlp.'+name+'.weight',dims,offset+l,.04)
p='model.layers.0.linear_attn.'
for name,dims,offset in [('in_proj_qkv.weight',[3,8],1),('in_proj_z.weight',[3,4],4),('in_proj_b.weight',[3,2],7),('in_proj_a.weight',[3,2],10),('conv1d.weight',[2,1,8],13),('A_log',[2],16),('dt_bias',[2],19),('norm.weight',[2],22),('out_proj.weight',[4,3],25)]:add(p+name,dims,offset,.04)
p='model.layers.1.self_attn.'
for name,dims,offset in [('q_proj.weight',[3,24],3),('k_proj.weight',[3,6],6),('v_proj.weight',[3,6],9),('o_proj.weight',[12,3],12),('q_norm.weight',[6],15),('k_norm.weight',[6],18)]:add(p+name,dims,offset,.03)
add('model.norm.weight',[3],5)
add('lm_head.weight',[3,4],11,.07)
history=[0.0]*8;state=[0.0]*8;keys=[];values=[];records={}
def ffn(x,l):
    p=f'model.layers.{l}.mlp.'
    prob=softmax(mv(w(p+'gate.weight'),x));y=[0.0]*3
    for e in range(2):
        a=mv(w(p+'experts.gate_proj.weight')[e*6:e*6+6],x)
        b=mv(w(p+'experts.up_proj.weight')[e*6:e*6+6],x)
        z=mv(w(p+'experts.down_proj.weight')[e*6:e*6+6],[silu(v)*u for v,u in zip(a,b)])
        y=[v+prob[e]*u for v,u in zip(y,z)]
    a=mv(w(p+'shared_expert.gate_proj.weight'),x);b=mv(w(p+'shared_expert.up_proj.weight'),x)
    z=mv(w(p+'shared_expert.down_proj.weight'),[silu(v)*u for v,u in zip(a,b)])
    g=1/(1+math.exp(-sum(a*b for a,b in zip(w(p+'shared_expert_gate.weight'),x))))
    return [v+g*u for v,u in zip(y,z)]
for token in range(2):
    x=series(3,2+token*5,.13);records[f't{token}_input']=x[:]
    for l in range(2):
        prefix=f'model.layers.{l}.';n=norm(x,w(prefix+'input_layernorm.weight'))
        if l==0:
            p=prefix+'linear_attn.'
            raw=mv(w(p+'in_proj_qkv.weight'),n);z=mv(w(p+'in_proj_z.weight'),n)
            a=mv(w(p+'in_proj_a.weight'),n);b=mv(w(p+'in_proj_b.weight'),n)
            taps=w(p+'conv1d.weight');c=[silu(history[i]*taps[2*i]+raw[i]*taps[2*i+1]) for i in range(8)];history=raw[:]
            q=[v/math.sqrt(2) for v in l2(c[:2])];k=l2(c[2:4]);v=c[4:];y=[]
            for h in range(2):
                decay=math.exp(-math.exp(w(p+'A_log')[h])*math.log1p(math.exp(a[h]+w(p+'dt_bias')[h])))
                d=[v*decay for v in state[h*4:h*4+4]]
                delta=[(v[h*2+j]-sum(k[i]*d[i*2+j] for i in range(2)))/(1+math.exp(-b[h])) for j in range(2)]
                s=[d[i*2+j]+k[i]*delta[j] for i in range(2) for j in range(2)];state[h*4:h*4+4]=s
                head=[sum(q[i]*s[i*2+j] for i in range(2)) for j in range(2)]
                y.extend(v/math.sqrt(sum(u*u for u in head)/2+1e-6)*w(p+'norm.weight')[j]*silu(z[h*2+j]) for j,v in enumerate(head))
            y=mv(w(p+'out_proj.weight'),y)
        else:
            p=prefix+'self_attn.';projected=mv(w(p+'q_proj.weight'),n)
            qs=[rope(norm(projected[h*12:h*12+6],w(p+'q_norm.weight')),token) for h in range(2)]
            keys.append(rope(norm(mv(w(p+'k_proj.weight'),n),w(p+'k_norm.weight')),token));values.append(mv(w(p+'v_proj.weight'),n));y=[]
            for h,q in enumerate(qs):
                prob=softmax([sum(a*b for a,b in zip(q,k))/math.sqrt(6) for k in keys])
                y.extend(sum(prob[t]*values[t][j] for t in range(token+1))/(1+math.exp(-projected[h*12+6+j])) for j in range(6))
            y=mv(w(p+'o_proj.weight'),y)
        x=[a+b for a,b in zip(x,y)];records[f't{token}_attention{l}']=x[:]
        y=ffn(norm(x,w(prefix+'post_attention_layernorm.weight')),l)
        x=[a+b for a,b in zip(x,y)];records[f't{token}_ffn{l}']=x[:]
    records[f't{token}_history']=history[:];records[f't{token}_state']=state[:]
    records[f't{token}_kv']=sum(keys,[])+[0.]*(6*(1-token))+sum(values,[])+[0.]*(6*(1-token))
    x=norm(x,w('model.norm.weight'));records[f't{token}_final']=x[:];records[f't{token}_logits']=mv(w('lm_head.weight'),x)
lines=['// Independent float64 scalar oracle. See generate_qwen_model_reference.py.']
lines.append('pub const WEIGHTS: &[(&str, &[usize], &[f32])] = &[')
for name,(dims,a) in weights.items():lines.append('('+json.dumps(name)+', &'+str(dims)+', &['+', '.join(f'{v:.17e}_f32' for v in a)+']),')
lines.append('];')
for name,a in records.items():lines.append(f'pub const {name.upper()}: [f32; {len(a)}] = ['+', '.join(f'{v:.17e}_f32' for v in a)+'];')
artifacts={'qwen_model_fixture.rs':'\n'.join(lines)+'\n','qwen_model_reference.json':json.dumps(dict(weights=weights,records=records),indent=2)+'\n'}
assert sum(len(v.encode()) for v in artifacts.values())<100000
for name,payload in artifacts.items():(out/name).write_bytes(payload.encode('utf8'))
print(json.dumps({name:hashlib.sha256(v.encode()).hexdigest() for name,v in artifacts.items()}))
