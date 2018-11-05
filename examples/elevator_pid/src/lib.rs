#![allow(dead_code)]

#[macro_use]
extern crate serde_derive;

use control_sim::units as si;
use control_sim::util::clamp;
use control_sim::*;

#[derive(Copy, Clone, Debug)]
enum LoopState {
    Unitialized,
    Zeroing,
    Running,
}

#[derive(Clone, Debug)]
struct ElevatorPIDLoop {
    state: LoopState,
    sp: si::Meter<f64>,
    last_err: si::Meter<f64>,
    zero_offset: si::Meter<f64>,
    zero_goal: si::Meter<f64>,
}

impl ElevatorPIDLoop {
    pub const ZEROING_SPEED: si::MeterPerSecond<f64> = const_unit!(0.04);
    pub const MAX_HEIGHT: si::Meter<f64> = const_unit!(2.5);
    pub const MIN_HEIGHT: si::Meter<f64> = const_unit!(-0.02);
    pub const DT: si::Second<f64> = const_unit!(1. / 200.);
    pub const KP: si::VoltPerMeter<f64> = const_unit!(20.0);
    pub const KD: units::VoltSecondPerMeter<f64> = const_unit!(5.);

    pub fn new() -> Self {
        Self {
            state: LoopState::Unitialized,
            last_err: 0. * si::M,
            sp: 0. * si::M,
            zero_offset: 0. * si::M,
            zero_goal: 0. * si::M,
        }
    }

    pub fn acc(volt: si::Volt<f64>, vel: si::MeterPerSecond<f64>) -> si::MeterPerSecond2<f64> {
        #![allow(non_snake_case)]
        let m = 5. * si::KG;
        let r = 0.1524 * si::M;
        let R = 12. * si::V / (133. * si::A);
        let G = 20.; // how much slower the output is than input
        let Kt = 24. * si::N * si::M / (133. * si::A);
        let Kv = (558.15629415 /*rad*/ / si::S) / (12. * si::V);

        (G * Kt * (Kv * volt * r - G * vel)) / (m * Kv * R * r * r)
    }
}

impl ElevatorPIDLoop {
    fn iterate(&mut self, encoder: si::Meter<f64>, limit: bool) -> si::Volt<f64> {
        let filtered_goal;
        match self.state {
            LoopState::Unitialized => {
                self.zero_goal = encoder;
                self.state = LoopState::Zeroing;
                return self.iterate(encoder, limit);
            }
            LoopState::Zeroing => {
                if limit {
                    self.state = LoopState::Running;
                    self.zero_offset = encoder;
                    self.last_err = 0. * si::M;
                    return self.iterate(encoder, limit);
                }
                self.zero_goal = self.zero_goal - (Self::ZEROING_SPEED * Self::DT);
                filtered_goal = self.zero_goal;
            }
            LoopState::Running => {
                filtered_goal = clamp(self.sp, Self::MIN_HEIGHT, Self::MAX_HEIGHT);
            }
        };
        let err = filtered_goal - (encoder - self.zero_offset);
        let v = clamp(
            err * Self::KP + ((err - self.last_err) / Self::DT) * Self::KD,
            -12. * si::V,
            12. * si::V,
        );

        self.last_err = err;
        return v;
    }

    fn set_goal(&mut self, sp: si::Meter<f64>) {
        self.sp = sp;
    }

    fn get_goal(&self) -> si::Meter<f64> {
        self.sp
    }

    fn state(&self) -> LoopState {
        self.state
    }
}

#[derive(Debug, Copy, Clone)]
struct ElevatorPhysicsState {
    pos: si::Meter<f64>,
    vel: si::MeterPerSecond<f64>,
}

#[derive(Debug, Copy, Clone, Serialize)]
struct ElevatorLog {
    time: f64,
    pos: f64,
    vel: f64,
    volts: f64,
    sp: f64,
}

impl HarnessAble for ElevatorPIDLoop {
    type State = ElevatorPhysicsState;
    type ControlResponse = si::Volt<f64>;
    type LogData = ElevatorLog;
    // 1000 sims per dt
    const SIMUL_DT: si::Second<f64> = const_unit!(1. / 200. / 1000.);
    const CONTROL_DT: si::Second<f64> = const_unit!(1. / 200.);
    fn sim_time(s: Self::State, r: Self::ControlResponse, dur: si::Second<f64>) -> Self::State {
        let mut elapsed = 0. * si::S;
        let mut pos = s.pos;
        let mut vel = s.vel;
        while elapsed < dur {
            vel += ElevatorPIDLoop::acc(r, vel) * Self::SIMUL_DT;
            pos += vel * Self::SIMUL_DT;
            elapsed += Self::SIMUL_DT;
        }
        ElevatorPhysicsState { pos, vel }
    }
}

#[derive(Debug, Clone)]
struct ElevatorShim {
    enc_off: si::Meter<f64>,
    control: ElevatorPIDLoop,
}

impl ElevatorShim {
    pub fn new(offset: si::Meter<f64>, control: ElevatorPIDLoop) -> Self {
        Self {
            enc_off: offset,
            control,
        }
    }

    pub fn controller(&self) -> &ElevatorPIDLoop {
        &self.control
    }

    pub fn controller_mut(&mut self) -> &mut ElevatorPIDLoop {
        &mut self.control
    }
}

impl StateShim<ElevatorPIDLoop> for ElevatorShim {
    fn update(&mut self, state: ElevatorPhysicsState) -> si::Volt<f64> {
        self.control
            .iterate(state.pos + self.enc_off, state.pos <= 0. * si::M)
    }

    fn log_dat(
        &mut self,
        s: ElevatorPhysicsState,
        r: si::Volt<f64>,
        t: si::Second<f64>,
    ) -> ElevatorLog {
        ElevatorLog {
            pos: *(s.pos / si::M),
            vel: *(s.vel / si::MPS),
            sp: *(self.control.get_goal() / si::M),
            volts: *(r / si::V),
            time: *(t / si::S),
        }
    }

    fn assert(&mut self, state: ElevatorPhysicsState) {
        assert!(state.pos <= ElevatorPIDLoop::MAX_HEIGHT);
        assert!(state.pos >= ElevatorPIDLoop::MIN_HEIGHT);
        dbg_isfinite!(state.pos / si::M);
        dbg_isfinite!(state.vel / si::MPS);
        // etc.
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn with_harness() {
        let mut harness = SimulationHarness::new(
            ElevatorShim::new(1. * si::M, ElevatorPIDLoop::new()),
            ElevatorPhysicsState {
                pos: 0.1 * si::M,
                vel: 0. * si::MPS,
            },
            20,
        );
        harness.use_csv("harness.csv");
        harness.shim_mut().controller_mut().set_goal(1. * si::M);
        harness.run_time(30. * si::S);
    }
}
