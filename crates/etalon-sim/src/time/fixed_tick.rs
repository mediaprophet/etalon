//! Deterministic Fixed-Tick Multi-Agent Simulation Engine.
//!
//! Provides a strictly reproducible, zero-float-drift fixed-step simulation kernel.
//! Designed for:
//! - Multi-agent RTS and simulation worlds
//! - POET step-by-step document and macro automation
//! - Attestable distributed simulation replay across native and WASM
//!
//! Rules (AGENTS.md):
//! - Zero heap inside hot tick evaluation loops (Tier 1 hot-path).
//! - Fixed-point / integer arithmetic for coordinates and resource tallies.
//! - Seeded deterministic PRNG (PCG / xorshift).
//! - Strict command ordering: (target_tick, sequence, actor_did).
//! - Typed accepted / rejected receipts for all submitted intents.

use super::prng::DeterministicPrng;

/// Simulation frequency (20 Hz = 50ms per tick).
pub const SIMULATION_TICK_RATE_HZ: u32 = 20;
pub const SIMULATION_TICK_MS: u64 = 1000 / SIMULATION_TICK_RATE_HZ as u64;

/// Default capacity constants.
pub const MAX_ENTITIES: usize = 64;
pub const MAX_COMMANDS_PER_TICK: usize = 128;

/// 2D vector in fixed-point millimeter space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FixedVec2 {
    pub x_mm: i64,
    pub y_mm: i64,
}

impl FixedVec2 {
    #[inline]
    pub const fn new(x_mm: i64, y_mm: i64) -> Self {
        Self { x_mm, y_mm }
    }

    #[inline]
    pub fn to_meters_f32(self) -> (f32, f32) {
        (self.x_mm as f32 * 0.001, self.y_mm as f32 * 0.001)
    }

    #[inline]
    pub fn from_meters_f32(x_m: f32, y_m: f32) -> Self {
        Self {
            x_mm: (x_m * 1000.0) as i64,
            y_mm: (y_m * 1000.0) as i64,
        }
    }
}

/// Strongly typed command payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommandPayload {
    Move { dx_mm: i64, dy_mm: i64 },
    Harvest { resource_gain: u64 },
    SetPosition { x_mm: i64, y_mm: i64 },
    Custom { opcode: u16, arg0: i64, arg1: i64 },
}

/// Simulation command intent submitted by an agent or user.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SimulationCommand {
    pub target_tick: u64,
    pub sequence: u32,
    pub actor_did: u64,
    pub command_opcode: u16,
    pub target_entity: u64,
    pub arg0: i64,
    pub arg1: i64,
}

/// Ergonomic alias for `SimulationCommand`.
pub type SimCommand = SimulationCommand;

impl SimulationCommand {
    /// Constructs a command with a typed payload.
    pub fn with_payload(
        target_tick: u64,
        sequence: u32,
        actor_did: u64,
        target_entity: u64,
        payload: CommandPayload,
    ) -> Self {
        let (command_opcode, arg0, arg1) = match payload {
            CommandPayload::Move { dx_mm, dy_mm } => (0x01, dx_mm, dy_mm),
            CommandPayload::Harvest { resource_gain } => (0x02, resource_gain as i64, 0),
            CommandPayload::SetPosition { x_mm, y_mm } => (0x03, x_mm, y_mm),
            CommandPayload::Custom { opcode, arg0, arg1 } => (opcode, arg0, arg1),
        };
        Self {
            target_tick,
            sequence,
            actor_did,
            command_opcode,
            target_entity,
            arg0,
            arg1,
        }
    }

    /// Hash of the tick commands.
    pub fn command_hash(&self) -> u64 {
        let mut h = 0xcbf29ce484222325u64; // FNV-1a
        for &b in &self.target_tick.to_le_bytes() {
            h = (h ^ (b as u64)).wrapping_mul(0x100000001b3);
        }
        for &b in &self.sequence.to_le_bytes() {
            h = (h ^ (b as u64)).wrapping_mul(0x100000001b3);
        }
        for &b in &self.actor_did.to_le_bytes() {
            h = (h ^ (b as u64)).wrapping_mul(0x100000001b3);
        }
        for &b in &self.command_opcode.to_le_bytes() {
            h = (h ^ (b as u64)).wrapping_mul(0x100000001b3);
        }
        for &b in &self.target_entity.to_le_bytes() {
            h = (h ^ (b as u64)).wrapping_mul(0x100000001b3);
        }
        for &b in &self.arg0.to_le_bytes() {
            h = (h ^ (b as u64)).wrapping_mul(0x100000001b3);
        }
        for &b in &self.arg1.to_le_bytes() {
            h = (h ^ (b as u64)).wrapping_mul(0x100000001b3);
        }
        h
    }
}

/// Reason for rejecting a proposed command.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RejectionReason {
    InvalidTick = 1,
    UnauthorizedActor = 2,
    EntityNotFound = 3,
    InsufficientResources = 4,
    CollisionObstacle = 5,
    InvalidOpcode = 6,
}

/// Alias for `RejectionReason`.
pub type CommandReceiptStatus = RejectionReason;

/// Deterministic receipt issued for every executed command.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommandReceipt {
    Accepted {
        tick: u64,
        command_hash: u64,
        state_delta_hash: u64,
    },
    Rejected {
        tick: u64,
        command_hash: u64,
        reason: RejectionReason,
    },
}

/// Compact entity state in fixed-point space (millimeter coordinates).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AgentState {
    pub entity_id: u64,
    pub pos_x_mm: i64,
    pub pos_y_mm: i64,
    pub resource_tally: u64,
    pub task_opcode: u16,
    pub health_points: u16,
}

/// Ergonomic alias for `AgentState`.
pub type EntityState = AgentState;

/// Fixed-capacity multi-agent simulation world kernel (Zero Heap hot path).
pub struct FixedTickWorld<const MAX_AGENTS: usize, const MAX_QUEUE: usize> {
    pub seed: u64,
    pub prng: DeterministicPrng,
    pub current_tick: u64,
    pub agents: [AgentState; MAX_AGENTS],
    pub agent_count: usize,
    command_queue: [Option<SimulationCommand>; MAX_QUEUE],
    queue_len: usize,
}

/// Ergonomic alias for `FixedTickWorld`.
pub type WorldState<const MAX_AGENTS: usize, const MAX_QUEUE: usize> = FixedTickWorld<MAX_AGENTS, MAX_QUEUE>;

impl<const MAX_AGENTS: usize, const MAX_QUEUE: usize> FixedTickWorld<MAX_AGENTS, MAX_QUEUE> {
    /// Creates a default `FixedTickWorld`.
    pub fn new(seed: u64) -> Self {
        Self {
            seed,
            prng: DeterministicPrng::new(seed),
            current_tick: 0,
            agents: [AgentState {
                entity_id: 0,
                pos_x_mm: 0,
                pos_y_mm: 0,
                resource_tally: 0,
                task_opcode: 0,
                health_points: 100,
            }; MAX_AGENTS],
            agent_count: 0,
            command_queue: [None; MAX_QUEUE],
            queue_len: 0,
        }
    }

    /// Spawn or register an agent entity in the world.
    pub fn register_agent(&mut self, entity_id: u64, x_mm: i64, y_mm: i64) -> bool {
        if self.agent_count >= MAX_AGENTS {
            return false;
        }
        self.agents[self.agent_count] = AgentState {
            entity_id,
            pos_x_mm: x_mm,
            pos_y_mm: y_mm,
            resource_tally: 0,
            task_opcode: 0,
            health_points: 100,
        };
        self.agent_count += 1;
        true
    }

    /// Submit a command intent into the deterministic tick queue.
    pub fn submit_command(&mut self, cmd: SimulationCommand) -> Result<(), RejectionReason> {
        if cmd.target_tick < self.current_tick {
            return Err(RejectionReason::InvalidTick);
        }
        if self.queue_len >= MAX_QUEUE {
            return Err(RejectionReason::InsufficientResources);
        }
        self.command_queue[self.queue_len] = Some(cmd);
        self.queue_len += 1;
        Ok(())
    }

    /// Step the simulation by exactly one fixed tick (Zero Heap Tier 1 loop).
    /// Writes output receipts into the caller-supplied slice.
    pub fn step_tick(&mut self, out_receipts: &mut [CommandReceipt]) -> usize {
        self.current_tick += 1;
        let mut receipt_idx = 0;

        // 1. Process commands scheduled for this tick
        let mut i = 0;
        while i < self.queue_len {
            if let Some(cmd) = self.command_queue[i] {
                if cmd.target_tick == self.current_tick {
                    let receipt = self.execute_command(&cmd);
                    if receipt_idx < out_receipts.len() {
                        out_receipts[receipt_idx] = receipt;
                        receipt_idx += 1;
                    }
                    // Remove executed command by swapping with last
                    self.command_queue[i] = self.command_queue[self.queue_len - 1];
                    self.command_queue[self.queue_len - 1] = None;
                    self.queue_len -= 1;
                    continue; // Re-check at index i
                }
            }
            i += 1;
        }

        // 2. Deterministic ambient world update (passive resource regeneration / motion)
        for a in 0..self.agent_count {
            let rnd = self.prng.next_bounded(4);
            match rnd {
                0 => self.agents[a].pos_x_mm += 10,
                1 => self.agents[a].pos_x_mm -= 10,
                2 => self.agents[a].pos_y_mm += 10,
                _ => self.agents[a].pos_y_mm -= 10,
            }
        }

        receipt_idx
    }

    fn execute_command(&mut self, cmd: &SimulationCommand) -> CommandReceipt {
        let cmd_hash = cmd.command_hash();

        // Find target agent
        let mut agent_idx = None;
        for a in 0..self.agent_count {
            if self.agents[a].entity_id == cmd.target_entity {
                agent_idx = Some(a);
                break;
            }
        }

        let idx = match agent_idx {
            Some(i) => i,
            None => {
                return CommandReceipt::Rejected {
                    tick: self.current_tick,
                    command_hash: cmd_hash,
                    reason: RejectionReason::EntityNotFound,
                };
            }
        };

        // Opcode dispatch:
        // 0x01 = Move (arg0 = delta_x_mm, arg1 = delta_y_mm)
        // 0x02 = Harvest (arg0 = resource_gain)
        // 0x03 = SetPosition (arg0 = x_mm, arg1 = y_mm)
        match cmd.command_opcode {
            0x01 => {
                self.agents[idx].pos_x_mm += cmd.arg0;
                self.agents[idx].pos_y_mm += cmd.arg1;
                CommandReceipt::Accepted {
                    tick: self.current_tick,
                    command_hash: cmd_hash,
                    state_delta_hash: (cmd.arg0 as u64) ^ (cmd.arg1 as u64),
                }
            }
            0x02 => {
                if cmd.arg0 < 0 {
                    return CommandReceipt::Rejected {
                        tick: self.current_tick,
                        command_hash: cmd_hash,
                        reason: RejectionReason::InsufficientResources,
                    };
                }
                self.agents[idx].resource_tally = self.agents[idx]
                    .resource_tally
                    .saturating_add(cmd.arg0 as u64);
                CommandReceipt::Accepted {
                    tick: self.current_tick,
                    command_hash: cmd_hash,
                    state_delta_hash: cmd.arg0 as u64,
                }
            }
            0x03 => {
                self.agents[idx].pos_x_mm = cmd.arg0;
                self.agents[idx].pos_y_mm = cmd.arg1;
                CommandReceipt::Accepted {
                    tick: self.current_tick,
                    command_hash: cmd_hash,
                    state_delta_hash: (cmd.arg0 as u64) ^ (cmd.arg1 as u64),
                }
            }
            _ => CommandReceipt::Rejected {
                tick: self.current_tick,
                command_hash: cmd_hash,
                reason: RejectionReason::InvalidOpcode,
            },
        }
    }

    /// Compute a canonical 64-bit state fingerprint across all entities.
    pub fn compute_state_hash(&self) -> u64 {
        let mut h = 0x811c9dc5u64 ^ self.current_tick;
        for a in 0..self.agent_count {
            h ^= self.agents[a].entity_id.wrapping_mul(0x100000001b3);
            h ^= (self.agents[a].pos_x_mm as u64).wrapping_mul(0x100000001b3);
            h ^= (self.agents[a].pos_y_mm as u64).wrapping_mul(0x100000001b3);
            h ^= self.agents[a].resource_tally.wrapping_mul(0x100000001b3);
        }
        h
    }

    /// Lookup current coordinates of an agent (in millimeters).
    pub fn get_agent_position(&self, entity_id: u64) -> Option<(i64, i64)> {
        for a in 0..self.agent_count {
            if self.agents[a].entity_id == entity_id {
                return Some((self.agents[a].pos_x_mm, self.agents[a].pos_y_mm));
            }
        }
        None
    }

    /// Set position directly for an agent.
    pub fn set_agent_position(&mut self, entity_id: u64, x_mm: i64, y_mm: i64) -> bool {
        for a in 0..self.agent_count {
            if self.agents[a].entity_id == entity_id {
                self.agents[a].pos_x_mm = x_mm;
                self.agents[a].pos_y_mm = y_mm;
                return true;
            }
        }
        false
    }

    /// Spatial box selection / range query (Zero Heap Tier 1 hot path).
    ///
    /// Writes matching `entity_id` values into `out_entity_ids` and returns the count.
    pub fn query_agents_in_bounds(
        &self,
        min_x_mm: i64,
        min_y_mm: i64,
        max_x_mm: i64,
        max_y_mm: i64,
        out_entity_ids: &mut [u64],
    ) -> usize {
        let mut count = 0;
        for a in 0..self.agent_count {
            let px = self.agents[a].pos_x_mm;
            let py = self.agents[a].pos_y_mm;
            if px >= min_x_mm && px <= max_x_mm && py >= min_y_mm && py <= max_y_mm {
                if count < out_entity_ids.len() {
                    out_entity_ids[count] = self.agents[a].entity_id;
                    count += 1;
                } else {
                    break;
                }
            }
        }
        count
    }
}
