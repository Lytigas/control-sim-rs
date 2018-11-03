#![feature(test)]
#![feature(const_fn)]

pub trait HarnessAble {
    /// A type representing the state of the system
    type State: Copy;
    /// A type that represents the output of the controller
    type ControlResponse: Copy;
    /// Physically simulates the system over the time where the control response is constant
    fn sim_time(s: Self::State, r: Self::ControlResponse, dur: si::Second<f64>) -> Self::State;
    /// The duration of one period for physics simulation
    const SIMUL_DT: si::Second<f64>;
    /// The interval between control response updates
    const CONTROL_DT: si::Second<f64>;
}

/// Shims a physically simulated state into simulated sensors passed to the control loop.
///
/// Can be used to simulate things like encoder offsets, failing sensors.
pub trait StateShim<SYS>
where
    SYS: HarnessAble,
{
    /// The output of the shimmed controller based on the current physical state
    fn update(&mut self, SYS::State) -> SYS::ControlResponse;

    /// Include safety assertions about the physical state. Will be called in tests
    fn assert(&mut self, SYS::State) {}
}

pub struct SimulationHarness<SYS, SHIM>
where
    SYS: HarnessAble,
    SHIM: StateShim<SYS>,
{
    shim: SHIM,
    state: SYS::State,
    time: si::Second<f64>,
    log_every: u32,
}

impl<SYS, SHIM> SimulationHarness<SYS, SHIM>
where
    SYS: HarnessAble,
    SHIM: StateShim<SYS>,
{
    pub fn new(shim: SHIM, initial: SYS::State, log_every: u32) -> Self {
        Self {
            shim,
            state: initial,
            time: 0. * si::S,
            log_every,
        }
    }

    pub fn run_time(&mut self, time: si::Second<f64>) -> SYS::State {
        let mut elapsed = 0. * si::S;
        while elapsed < time {
            let response = self.shim.update(self.state);
            self.state = SYS::sim_time(self.state, response, SYS::CONTROL_DT);
            self.shim.assert(self.state);
            elapsed += SYS::CONTROL_DT;
            self.time += SYS::CONTROL_DT;
        }
        return self.state;
    }
}

#[cfg(test)]
#[macro_use]
extern crate assert_approx_eq;

extern crate alga;
#[macro_use]
extern crate dimensioned as dim;
use dim::si;

pub mod integration;

pub trait Simulatable {
    type State;
    const SIM_DT: si::Second<f64>;
    fn sim_duration(&mut self, now: Self::State, dur: si::Second<f64>) -> Self::State;
}

pub trait SimulationLaw<V> {
    fn acc(volt: si::Volt<V>, vel: si::MeterPerSecond<V>) -> si::MeterPerSecond2<V>;
}

#[macro_use]
mod util {
    pub fn clamp<T: PartialOrd>(a: T, lower: T, upper: T) -> T {
        debug_assert!(lower < upper);
        if a < lower {
            lower
        } else if a > upper {
            upper
        } else {
            a
        }
    }

    use dim::si;
    pub trait DistanceSensor<V> {
        fn get(&self) -> si::Meter<V>;
    }

    pub trait LimitSwitch {
        fn get(&self) -> bool;
    }

    macro_rules! const_unit {
        ($val:expr) => {
            dim::si::SI {
                value_unsafe: $val,
                _marker: ::std::marker::PhantomData,
            }
        };
    }

    pub trait Polarity {
        fn is_invert() -> bool;
    }
    struct Forward;
    impl Polarity for Forward {
        #[inline]
        fn is_invert() -> bool {
            false
        }
    }
    struct Reverse;
    impl Polarity for Reverse {
        #[inline]
        fn is_invert() -> bool {
            true
        }
    }
}
#[macro_use]
mod assertions;

mod example {
    use super::*;
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

    use dim::typenum::{N1, N2, P1, Z0};
    pub type VoltSecondPerMeter<V> = si::SI<V, tarr![P1, P1, N2, N1, Z0, Z0, Z0]>; // also Newtons per Amp

    impl ElevatorPIDLoop {
        pub const ZEROING_SPEED: si::MeterPerSecond<f64> = const_unit!(0.01);
        pub const MAX_HEIGHT: si::Meter<f64> = const_unit!(2.5);
        pub const MIN_HEIGHT: si::Meter<f64> = const_unit!(-0.02);
        pub const DT: si::Second<f64> = const_unit!(1. / 200.);
        pub const KP: si::VoltPerMeter<f64> = const_unit!(20.0);
        pub const KD: VoltSecondPerMeter<f64> = const_unit!(5.);

        fn new() -> Self {
            Self {
                state: LoopState::Unitialized,
                last_err: 0. * si::M,
                sp: 0. * si::M,
                zero_offset: 0. * si::M,
                zero_goal: 0. * si::M,
            }
        }
    }

    #[derive(Copy, Clone, Debug)]
    struct ElevatorState {
        pos: si::Meter<f64>,
        vel: si::MeterPerSecond<f64>,
        vol: si::Volt<f64>,
    }

    impl ElevatorState {
        pub fn get_encoder(&self, offset: si::Meter<f64>) -> si::Meter<f64> {
            self.pos + offset
        }

        pub fn get_limit(&self) -> bool {
            self.pos <= 0.0 * si::M
        }
    }

    impl Simulatable for ElevatorPIDLoop {
        type State = ElevatorState;
        const SIM_DT: si::Second<f64> = const_unit!(1. / 200. / 100.);
        fn sim_duration(&mut self, now: Self::State, dur: si::Second<f64>) -> Self::State {
            let mut elapsed = 0. * si::S;
            let mut pos = now.pos;
            let mut vel = now.vel;
            while elapsed < dur {
                vel += ElevatorSim::acc(now.vol, vel) * Self::SIM_DT;
                pos += vel * Self::SIM_DT;
                elapsed += Self::SIM_DT;
            }
            ElevatorState {
                pos,
                vel,
                vol: now.vol,
            }
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
                    dbg_lt!(encoder - self.zero_offset, Self::MAX_HEIGHT);
                    dbg_gt!(encoder - self.zero_offset, Self::MIN_HEIGHT);
                    filtered_goal = util::clamp(self.sp, Self::MIN_HEIGHT, Self::MAX_HEIGHT);
                }
            };
            let err = filtered_goal - (encoder - self.zero_offset);
            let v = util::clamp(
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

    impl HarnessAble for ElevatorPIDLoop {
        type State = ElevatorPhysicsState;
        type ControlResponse = si::Volt<f64>;
        // 1000 sims per dt
        const SIMUL_DT: si::Second<f64> = const_unit!(1. / 200. / 1000.);
        const CONTROL_DT: si::Second<f64> = const_unit!(1. / 200.);
        fn sim_time(s: Self::State, r: Self::ControlResponse, dur: si::Second<f64>) -> Self::State {
            let mut elapsed = 0. * si::S;
            let mut pos = s.pos;
            let mut vel = s.vel;
            while elapsed < dur {
                vel += ElevatorSim::acc(r, vel) * Self::SIMUL_DT;
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
    }

    impl StateShim<ElevatorPIDLoop> for ElevatorShim {
        fn update(&mut self, state: ElevatorPhysicsState) -> si::Volt<f64> {
            self.control
                .iterate(state.pos + self.enc_off, state.pos <= 0. * si::M)
        }

        fn assert(&mut self, state: ElevatorPhysicsState) {
            assert!(state.pos <= ElevatorPIDLoop::MAX_HEIGHT);
            assert!(state.pos <= ElevatorPIDLoop::MIN_HEIGHT);
            dbg_isfinite!(state.pos / si::M);
            dbg_isfinite!(state.vel / si::MPS);
            // etc.
        }
    }

    struct ElevatorSim;
    impl SimulationLaw<f64> for ElevatorSim {
        fn acc(volt: si::Volt<f64>, vel: si::MeterPerSecond<f64>) -> si::MeterPerSecond2<f64> {
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

    #[cfg(test)]
    mod test {
        use super::*;
        #[test]
        fn basic() {
            let mut state = ElevatorState {
                pos: 0.1 * si::M,
                vel: 0.0 * si::MPS,
                vol: 0.0 * si::V,
            };
            let mut l = ElevatorPIDLoop::new();
            l.set_goal(1. * si::M);

            extern crate csv;
            let mut csv = csv::WriterBuilder::new()
                .delimiter(b'\t')
                .from_path("./test.csv")
                .unwrap();
            csv.write_byte_record(&csv::ByteRecord::from(vec![
                "pos", "vel", "volt", "setpoint",
            ]))
            .unwrap();
            let mut data = Vec::new();
            for i in 0..20000 {
                // 50 sec
                state.vol = l.iterate(state.get_encoder(-12.01 * si::M), state.get_limit());
                state = l.sim_duration(state, ElevatorPIDLoop::DT);
                assert!(state.pos <= ElevatorPIDLoop::MAX_HEIGHT);
                dbg_isfinite!(state.pos / si::M);
                dbg_isfinite!(state.vel / si::MPS);
                dbg_isfinite!(state.vol / si::V);
                if i % 40 == 0 {
                    data.push((
                        *(state.pos / si::M),
                        *(state.vel / si::MPS),
                        *(state.vol / si::V),
                        *(l.get_goal() / si::M),
                    ));
                }
            }
            data.iter().for_each(|r| csv.serialize(r).unwrap());
        }

        #[test]
        fn with_harness() {
            let harness = SimulationHarness::new(
                ElevatorShim::new(1. * si::M, ElevatorPIDLoop::new()),
                ElevatorPhysicsState {
                    pos: 0. * si::M,
                    vel: 0. * si::MPS,
                },
                20,
            );
            harness.run_time(10 * si::S),
        }
    }
}
