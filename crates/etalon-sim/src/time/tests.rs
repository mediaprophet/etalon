//! Unit Tests for Fixed-Tick Multi-Agent Simulation Engine.

use super::fixed_tick::*;

#[test]
fn test_fixed_tick_multi_agent_deterministic_replay() {
    let mut sim1 = FixedTickWorld::<16, 64>::new(0xDEAD_BEEF);
    let mut sim2 = FixedTickWorld::<16, 64>::new(0xDEAD_BEEF);

    sim1.register_agent(1001, 1000, 2000);
    sim1.register_agent(1002, 5000, 6000);

    sim2.register_agent(1001, 1000, 2000);
    sim2.register_agent(1002, 5000, 6000);

    let cmd = SimulationCommand {
        target_tick: 5,
        sequence: 1,
        actor_did: 0xAAAA,
        command_opcode: 0x01, // Move
        target_entity: 1001,
        arg0: 150,
        arg1: -200,
    };

    sim1.submit_command(cmd).unwrap();
    sim2.submit_command(cmd).unwrap();

    let mut receipts1 = [CommandReceipt::Rejected {
        tick: 0,
        command_hash: 0,
        reason: RejectionReason::InvalidTick,
    }; 8];
    let mut receipts2 = [CommandReceipt::Rejected {
        tick: 0,
        command_hash: 0,
        reason: RejectionReason::InvalidTick,
    }; 8];

    for _ in 0..100 {
        sim1.step_tick(&mut receipts1);
        sim2.step_tick(&mut receipts2);
    }

    // Both simulations must have 100% bit-identical state hashes after 100 ticks
    assert_eq!(sim1.compute_state_hash(), sim2.compute_state_hash());
    assert_eq!(sim1.agents[0].pos_x_mm, sim2.agents[0].pos_x_mm);
    assert_eq!(sim1.agents[0].pos_y_mm, sim2.agents[0].pos_y_mm);
}

#[test]
fn test_fixed_tick_rejected_command_leaves_state_intact() {
    let mut sim = FixedTickWorld::<8, 16>::new(42);
    sim.register_agent(500, 0, 0);

    // Submit command targeting non-existent entity
    let bad_cmd = SimulationCommand {
        target_tick: 1,
        sequence: 1,
        actor_did: 0x1,
        command_opcode: 0x01,
        target_entity: 9999, // Does not exist
        arg0: 100,
        arg1: 100,
    };
    sim.submit_command(bad_cmd).unwrap();

    let mut receipts = [CommandReceipt::Rejected {
        tick: 0,
        command_hash: 0,
        reason: RejectionReason::InvalidTick,
    }; 4];
    let n = sim.step_tick(&mut receipts);
    assert_eq!(n, 1);

    assert!(matches!(
        receipts[0],
        CommandReceipt::Rejected {
            reason: RejectionReason::EntityNotFound,
            ..
        }
    ));

    // State matches a world that advanced 1 tick with no mutations
    let mut baseline_sim = FixedTickWorld::<8, 16>::new(42);
    baseline_sim.register_agent(500, 0, 0);
    let mut baseline_receipts = [CommandReceipt::Rejected {
        tick: 0,
        command_hash: 0,
        reason: RejectionReason::InvalidTick,
    }; 4];
    baseline_sim.step_tick(&mut baseline_receipts);

    assert_eq!(sim.compute_state_hash(), baseline_sim.compute_state_hash());
    assert_eq!(sim.agents[0].pos_x_mm, baseline_sim.agents[0].pos_x_mm);
    assert_eq!(sim.agents[0].pos_y_mm, baseline_sim.agents[0].pos_y_mm);
    assert_eq!(
        sim.agents[0].resource_tally,
        baseline_sim.agents[0].resource_tally
    );
}

#[test]
fn test_fixed_tick_harvest_resource_tally() {
    let mut sim = FixedTickWorld::<4, 8>::new(12345);
    sim.register_agent(1, 0, 0);

    let harvest_cmd = SimulationCommand {
        target_tick: 1,
        sequence: 1,
        actor_did: 0x1,
        command_opcode: 0x02, // Harvest
        target_entity: 1,
        arg0: 250,
        arg1: 0,
    };
    sim.submit_command(harvest_cmd).unwrap();

    let mut receipts = [CommandReceipt::Rejected {
        tick: 0,
        command_hash: 0,
        reason: RejectionReason::InvalidTick,
    }; 4];
    sim.step_tick(&mut receipts);

    assert_eq!(sim.agents[0].resource_tally, 250);
    assert!(matches!(receipts[0], CommandReceipt::Accepted { .. }));
}

#[test]
fn test_spatial_query_and_position() {
    let mut sim = FixedTickWorld::<8, 16>::new(42);
    sim.register_agent(101, 1000, 2000);
    sim.register_agent(102, 5000, 5000);
    sim.register_agent(103, 1200, 2100);

    assert_eq!(sim.get_agent_position(101), Some((1000, 2000)));
    assert_eq!(sim.get_agent_position(999), None);

    let mut out_ids = [0u64; 4];
    let count = sim.query_agents_in_bounds(500, 1500, 2000, 2500, &mut out_ids);
    assert_eq!(count, 2);
    assert!(out_ids[..count].contains(&101));
    assert!(out_ids[..count].contains(&103));
}

#[test]
fn test_fixed_vec2_conversion() {
    let v = FixedVec2::from_meters_f32(1.5, -2.5);
    assert_eq!(v.x_mm, 1500);
    assert_eq!(v.y_mm, -2500);
    let (m_x, m_y) = v.to_meters_f32();
    assert!((m_x - 1.5).abs() < 1e-4);
    assert!((m_y - (-2.5)).abs() < 1e-4);
}

#[test]
fn test_sim_command_with_payload() {
    let cmd = SimulationCommand::with_payload(
        10,
        1,
        100,
        5,
        CommandPayload::Move {
            dx_mm: 50,
            dy_mm: -30,
        },
    );
    assert_eq!(cmd.command_opcode, 0x01);
    assert_eq!(cmd.arg0, 50);
    assert_eq!(cmd.arg1, -30);
}
