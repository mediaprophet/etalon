use naga::{
    BinaryOperator, Expression, Function, Handle, Literal, MathFunction, Module, Statement,
};

#[derive(Clone, Debug)]
enum Value {
    Scalar(f64),
    Vector(Vec<f64>),
    Matrix(Vec<Vec<f64>>),
    Aggregate(Vec<Value>),
}
impl Value {
    fn item(&self, index: usize) -> Self {
        match self {
            Self::Aggregate(v) => v[index].clone(),
            Self::Vector(v) => Self::Scalar(v[index]),
            Self::Matrix(v) => Self::Vector(v[index].clone()),
            _ => panic!("unsupported shader access: {self:?}"),
        }
    }
    fn vector(self) -> Vec<f64> {
        match self {
            Self::Vector(v) => v,
            _ => panic!("expected shader vector"),
        }
    }
}

fn multiply(a: Value, b: Value) -> Value {
    match (a, b) {
        (Value::Matrix(a), Value::Vector(b)) => Value::Vector(
            (0..a[0].len())
                .map(|r| a.iter().zip(&b).map(|(c, b)| c[r] * b).sum())
                .collect(),
        ),
        (Value::Matrix(a), Value::Matrix(b)) => Value::Matrix(
            b.into_iter()
                .map(|c| multiply(Value::Matrix(a.clone()), Value::Vector(c)).vector())
                .collect(),
        ),
        other => panic!("unsupported shader product: {other:?}"),
    }
}

fn evaluate(
    module: &Module,
    function: &Function,
    handle: Handle<Expression>,
    model: &Value,
    instance: &Value,
    normal: &[f64],
) -> Value {
    let eval = |h| evaluate(module, function, h, model, instance, normal);
    match &function.expressions[handle] {
        Expression::Literal(Literal::F32(v)) => Value::Scalar(f64::from(*v)),
        Expression::Literal(Literal::U32(v)) => Value::Scalar(f64::from(*v)),
        Expression::FunctionArgument(1) => Value::Vector(normal.to_vec()),
        Expression::FunctionArgument(4) => Value::Scalar(0.),
        Expression::GlobalVariable(h) => match module.global_variables[*h].name.as_deref() {
            Some("model") => model.clone(),
            Some("mesh_instances") => Value::Aggregate(vec![instance.clone()]),
            other => panic!("unexpected normal dependency: {other:?}"),
        },
        Expression::Load { pointer } => eval(*pointer),
        Expression::AccessIndex { base, index } => eval(*base).item(*index as usize),
        Expression::Access { base, index } => {
            let Value::Scalar(index) = eval(*index) else {
                panic!("invalid index")
            };
            eval(*base).item(index as usize)
        }
        Expression::Compose { components, .. } => Value::Vector(
            components
                .iter()
                .flat_map(|c| match eval(*c) {
                    Value::Scalar(v) => vec![v],
                    Value::Vector(v) => v,
                    _ => panic!("invalid vector component"),
                })
                .collect(),
        ),
        Expression::Binary {
            op: BinaryOperator::Multiply,
            left,
            right,
        } => multiply(eval(*left), eval(*right)),
        Expression::Swizzle {
            size,
            vector,
            pattern,
        } => {
            let v = eval(*vector).vector();
            Value::Vector(
                pattern[..*size as usize]
                    .iter()
                    .map(|p| v[*p as usize])
                    .collect(),
            )
        }
        Expression::Math {
            fun: MathFunction::Normalize,
            arg,
            ..
        } => {
            let v = eval(*arg).vector();
            let length = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            Value::Vector(v.into_iter().map(|x| x / length).collect())
        }
        other => panic!("unsupported production normal expression: {other:?}"),
    }
}

fn shader_normal(source: &str, model: &Value, instance: &Value, normal: &[f64]) -> Vec<f64> {
    let module = naga::front::wgsl::parse_str(source).unwrap();
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
    let function = &module
        .entry_points
        .iter()
        .find(|e| e.name == "vertex_main")
        .unwrap()
        .function;
    let result = function.result.as_ref().unwrap();
    let naga::TypeInner::Struct { members, .. } = &module.types[result.ty].inner else {
        panic!("vertex result")
    };
    let normal_index = members
        .iter()
        .position(|m| m.name.as_deref() == Some("normal"))
        .unwrap() as u32;
    for statement in &function.body {
        if let Statement::Store { pointer, value } = statement {
            if let Expression::AccessIndex { base, index } = &function.expressions[*pointer] {
                if *index == normal_index
                    && matches!(function.expressions[*base], Expression::LocalVariable(_))
                {
                    return evaluate(&module, function, *value, model, instance, normal).vector();
                }
            }
        }
    }
    panic!("production vertex normal store not found")
}

fn cross(a: &[f64], b: &[f64]) -> Vec<f64> {
    vec![
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn normalise(mut v: Vec<f64>) -> Vec<f64> {
    let length = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    for x in &mut v {
        *x /= length;
    }
    v
}
fn fixture(scale: [f64; 3]) -> (Value, Value, Vec<f64>, Vec<f64>) {
    let (s, c) = 0.6f64.sin_cos();
    let rotation = [vec![c, s, 0.], vec![-s, c, 0.], vec![0., 0., 1.]];
    let mut world: Vec<Vec<f64>> = rotation
        .iter()
        .enumerate()
        .map(|(i, c)| c.iter().map(|v| v * scale[i]).chain([0.]).collect())
        .collect();
    world.push(vec![7., -3., 2., 1.]);
    let inverse: Vec<Vec<f64>> = rotation
        .iter()
        .enumerate()
        .map(|(i, c)| c.iter().map(|v| v / scale[i]).collect())
        .collect();
    let (s, c) = 0.35f64.sin_cos();
    let model = Value::Matrix(vec![
        vec![1., 0., 0., 0.],
        vec![0., c, s, 0.],
        vec![0., -s, c, 0.],
        vec![0., 0., 0., 1.],
    ]);
    let a = vec![1., 2., 0.3];
    let b = vec![-0.2, 0.7, 1.5];
    let normal = normalise(cross(&a, &b));
    let transform = |mut v: Vec<f64>| {
        v.push(0.);
        multiply(
            model.clone(),
            multiply(Value::Matrix(world.clone()), Value::Vector(v)),
        )
        .vector()
    };
    let mut expected = normalise(cross(&transform(a), &transform(b)));
    for x in &mut expected {
        *x *= scale.iter().product::<f64>().signum();
    }
    (
        model,
        Value::Aggregate(vec![Value::Matrix(world), Value::Matrix(inverse)]),
        normal,
        expected,
    )
}

#[test]
fn actual_ao_vertex_normal_is_orthogonal_after_nonuniform_reflected_instances() {
    for scale in [[1., 1., 1.], [2., 3., 0.5], [-2., 3., 0.5], [0.1, 9., 2.]] {
        let (model, instance, normal, expected) = fixture(scale);
        let actual = shader_normal(
            include_str!("shaders/ao_prepass.wgsl"),
            &model,
            &instance,
            &normal,
        );
        assert!(
            actual
                .iter()
                .zip(expected)
                .all(|(a, b)| (a - b).abs() < 1e-7),
            "{scale:?}: {actual:?}"
        );
    }
}

#[test]
fn prior_world_matrix_normal_path_fails_the_independent_geometric_oracle() {
    let source = include_str!("shaders/ao_prepass.wgsl").replace(
        "(model * vec4<f32>(instance.normal_from_local * normal, 0.0)).xyz",
        "(instance_model * vec4<f32>(normal, 0.0)).xyz",
    );
    let (model, instance, normal, expected) = fixture([2., 3., 0.5]);
    let prior = shader_normal(&source, &model, &instance, &normal);
    assert!(
        prior
            .iter()
            .zip(expected)
            .map(|(a, b)| (a - b).abs())
            .sum::<f64>()
            > 0.5
    );
}
