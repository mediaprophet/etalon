# naga-types

Naga-types contains some types used by both naga and etalon. Naga may be an optional dependency of etalon in the future,
so these can't live in naga. Additionally, naga is a mostly independent crate, so it cannot depend on etalon-types.
For this reason, the types must live in a naga-specific crate but not naga itself. Naga-types serves that purpose.
