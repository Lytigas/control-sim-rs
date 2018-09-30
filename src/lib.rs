#![feature(test)]

#[cfg(test)]
#[macro_use]
extern crate assert_approx_eq;

extern crate alga;
#[macro_use]
extern crate dimensioned as dim;
use dim::si;

pub mod integration;

pub trait Simulatable<V> {
    type Sim: SimulationLaw<V>;
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
        pub const KP: si::VoltPerMeter<f64> = const_unit!(0.8);
        pub const KD: VoltSecondPerMeter<f64> = const_unit!(0.);

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

    impl Simulatable<f64> for ElevatorPIDLoop {
        type Sim = ElevatorSim;
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

        fn state(&self) -> LoopState {
            self.state
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
            let mut encoder = 0.3 * si::M;
            let mut limit = false;
            let mut v = 0. * si::MPS;
            let mut l = ElevatorPIDLoop::new();
            l.set_goal(1. * si::M);
            fn sim_time(
                l: &mut ElevatorPIDLoop,
                v: si::Volt<f64>,
                h: si::Second<f64>,
                vel: &mut si::MeterPerSecond<f64>,
                enc: &mut si::Meter<f64>,
                limit: &mut bool,
            ) {
                let dt = 1. / 10000. * si::S;
                assert!(*(h / dt) as u32 > 0);
                for _ in 0..(*(h / dt) as u32) {
                    let acc = <ElevatorPIDLoop as Simulatable<f64>>::Sim::acc(v, *vel);
                    *vel = *vel + acc * dt;
                    *enc = *enc + *vel * dt;
                    *limit = *enc <= (0.0 * si::M);
                }

                assert!(*enc < 2.5 * si::M);
            }
            extern crate csv;
            let mut csv = csv::Writer::from_path("./test.csv").unwrap();
            let mut vec = Vec::new();
            for i in 0..20000 {
                // 50 sec
                let volt = l.iterate(encoder + 13.01 * si::M, limit);
                sim_time(
                    &mut l,
                    volt,
                    ElevatorPIDLoop::DT,
                    &mut v,
                    &mut encoder,
                    &mut limit,
                );
                dbg_isfinite!(*(encoder / si::M));
                dbg_isfinite!(*(v / si::MPS));
                dbg_isfinite!(*(volt / si::V));
                if i % 40 == 0 {
                    vec.push((
                        //*(encoder / si::M + 13.01),
                        *(encoder / si::M),
                        *(v / si::MPS),
                        *(volt / si::V),
                        1.,
                    ))
                }
            }
            vec.iter().for_each(|r| csv.serialize(r).unwrap());
        }
    }
}
