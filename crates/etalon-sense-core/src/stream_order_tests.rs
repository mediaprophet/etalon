use super::*;

#[test]
fn gaps_are_visible_and_invalid_observations_do_not_advance() {
    let mut order = StreamOrder::new(1, 1).unwrap();
    let t = ClockStamp::new(1, 1, 10, 2).unwrap();
    assert_eq!(order.observe(1, 1, 4, t), Ok(SequenceEvent::First));
    assert_eq!(order.observe(1, 1, 4, t), Err(ContractError::InvalidSource));
    assert_eq!(order.observe(2, 1, 5, t), Err(ContractError::InvalidSource));
    assert_eq!(order.last().unwrap().0, 4);
    assert_eq!(order.observe(1, 1, 7, t), Ok(SequenceEvent::Gap { missing: 2 }));
    assert_eq!(order.observe(1, 1, 8, ClockStamp::new(1, 2, 11, 0).unwrap()), Err(ContractError::ClockDomainMismatch));
    assert_eq!(order.last().unwrap().0, 7);
}

#[test]
fn unknown_times_keep_sequence_order_and_known_clock_epoch_checks() {
    let mut order = StreamOrder::new(1, 1).unwrap();
    let first = ClockStamp::new(8, 4, 100, 1).unwrap();
    assert_eq!(order.observe_optional(1, 1, 5, None), Ok(SequenceEvent::First));
    assert_eq!(order.observe_optional(1, 1, 6, None), Ok(SequenceEvent::Contiguous));
    assert_eq!(order.last_observation(), Some((6, None)));
    assert_eq!(order.observe_optional(1, 1, 7, Some(first)), Ok(SequenceEvent::Contiguous));
    assert_eq!(order.observe_optional(1, 1, 8, None), Ok(SequenceEvent::Contiguous));
    assert_eq!(order.last_known_timestamp(), Some(first));
    assert_eq!(order.last_observation(), Some((8, None)));
    assert_eq!(
        order.observe_optional(1, 1, 9, Some(ClockStamp::new(8, 5, 101, 0).unwrap())),
        Err(ContractError::ClockDomainMismatch),
    );
    assert_eq!(
        order.observe_optional(1, 1, 9, Some(ClockStamp::new(8, 4, 99, 0).unwrap())),
        Err(ContractError::InvalidTime),
    );
    assert_eq!(order.observe_optional(1, 1, 8, None), Err(ContractError::InvalidSource));
    assert_eq!(order.last_observation(), Some((8, None)));
}
