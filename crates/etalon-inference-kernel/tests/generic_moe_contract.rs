use etalon_inference_kernel::moe::{dispatch::*, routing::*};
#[path = "support/qwen_moe_alloc.rs"]
mod allocation;

#[test]
fn configured_widths_select_upper_half_without_allocation() {
    let mut logits = [0.0; 512];
    for (id, logit) in logits.iter_mut().enumerate() {
        *logit = id as f32 / 512.0;
    }
    let mut ids = [999; MAX_MOE_TOPK];
    let mut weights = [-9.0; MAX_MOE_TOPK];
    for k in [6, 8, 10] {
        allocation::assert_zero_alloc("generic routing", || {
            assert_eq!(route_topk(&logits, k, true, &mut ids, &mut weights), Ok(k));
        });
        for (slot, &id) in ids[..k].iter().enumerate() {
            assert_eq!(id as usize, 511 - slot);
        }
        assert!((weights[..k].iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }
}

#[test]
fn dispatch_visits_configured_experts_including_511() {
    let mut router = [0.0; 512];
    for (i, weight) in router.iter_mut().enumerate() {
        *weight = i as f32;
    }
    for k in [6, 8, 10] {
        let mut visits = [0; MAX_MOE_TOPK];
        let mut count = 0;
        let mut out = [42.0];
        allocation::assert_zero_alloc("generic dispatch", || {
            dispatch_moe_step(
                &[1.0],
                1,
                &router,
                512,
                k,
                |expert| {
                    visits[count] = expert;
                    count += 1;
                    None
                },
                None,
                0.0,
                &mut out,
            )
            .unwrap();
        });
        assert_eq!(count, k);
        for (slot, &id) in visits[..count].iter().enumerate() {
            assert_eq!(id as usize, 511 - slot);
        }
    }
}

#[test]
fn malformed_routing_buffers_leave_outputs_untouched() {
    let mut ids = [99; 2];
    let mut weights = [98.0; 2];
    for (k, expected) in [(0, MoeError::InvalidTopK), (3, MoeError::BufferTooSmall)] {
        assert_eq!(
            route_topk(&[1.0; 10], k, true, &mut ids, &mut weights),
            Err(expected)
        );
    }
    assert_eq!(
        route_topk(&[1.0], 2, true, &mut ids, &mut weights),
        Err(MoeError::InsufficientExperts)
    );
    assert_eq!(ids, [99; 2]);
    assert_eq!(weights, [98.0; 2]);
}

#[test]
fn dispatch_rejects_short_router_and_zero_topk_before_fetch() {
    let mut out = [42.0];
    assert_eq!(
        dispatch_moe_step(
            &[1.0],
            1,
            &[1.0],
            512,
            10,
            |_| panic!("fetch"),
            None,
            0.0,
            &mut out
        ),
        Err(MoeDispatchError::InputOrAccumulatorTooSmall)
    );
    assert_eq!(
        dispatch_moe_step(
            &[1.0],
            1,
            &[1.0],
            1,
            0,
            |_| panic!("fetch"),
            None,
            0.0,
            &mut out
        ),
        Err(MoeDispatchError::Routing(MoeError::InvalidTopK))
    );
    assert_eq!(out, [42.0]);
    assert_eq!(MAX_ROUTED_EXPERTS, 512);
}
