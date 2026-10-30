use alloc::string::String;

use crate::{Binding, BuiltIn, Function, Module, TypeInner};

pub(super) const VIEW_INDEX_SEMANTIC: &str = "NAGA_VIEW_INDEX";

pub(super) struct ForwardedViewIndex {
    pub source: Option<ViewIndexSource>,
    pub name: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ViewIndexSource {
    pub argument: u32,
    pub member: Option<u32>,
}

pub(super) fn view_index_source(module: &Module, function: &Function) -> Option<ViewIndexSource> {
    for (argument, input) in function.arguments.iter().enumerate() {
        let member = match module.types[input.ty].inner {
            TypeInner::Struct { ref members, .. } => members
                .iter()
                .position(|member| is_view_index(&member.binding))
                .map(|index| Some(index as u32)),
            _ => is_view_index(&input.binding).then_some(None),
        };
        if let Some(member) = member {
            return Some(ViewIndexSource {
                argument: argument as u32,
                member,
            });
        }
    }
    None
}

pub(super) fn is_view_index(binding: &Option<Binding>) -> bool {
    *binding == Some(Binding::BuiltIn(BuiltIn::ViewIndex))
}
