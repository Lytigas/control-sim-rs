#![feature(test)]

#[cfg(test)]
#[macro_use]
extern crate assert_approx_eq;

extern crate alga;
use alga::general::Real;
use alga::linear::VectorSpace;

pub fn rk4_one_var(
    tn: f64,
    xn: f64,
    vn: f64,
    h: f64,
    acc_fun: impl Fn(f64, f64) -> f64,
) -> (f64, f64) {
    let k1 = acc_fun(tn, vn);
    let k2 = acc_fun(tn + h / 2., vn + k1 * h / 2.);
    let k3 = acc_fun(tn + h / 2., vn + k2 * h / 2.);
    let k4 = acc_fun(tn + h, vn + k3 * h);
    let vn1 = vn + (k1 + 2. * k2 + 2. * k3 + k4) * (h / 6.);

    let k1x = vn;
    let k2x = vn + k1x * h / 2.;
    let k3x = vn + k2x * h / 2.;
    let k4x = vn + k3x * h;
    let xn1 = xn + (k1x + 2. * k2x + 2. * k3x + k4x) * (h / 6.);

    return (xn1, vn1);
}

// https://scicomp.stackexchange.com/questions/21060/runge-kutta-simulation-for-projectile-motion-with-drag
pub fn rk4<S, V>(tn: S, xn: V, vn: V, h: S, acc_fun: impl Fn(S, V) -> V) -> (V, V)
where
    S: Real,
    V: VectorSpace<Field = S>,
{
    let _2 = S::from_i32(2).unwrap();
    let _6 = S::from_i32(6).unwrap();

    let __2 = _2.recip();
    let __6 = _6.recip();

    let k1 = acc_fun(tn, vn.clone());
    let k2 = acc_fun(tn + (h * __2), vn.clone() + k1.clone() * (h * __2));
    let k3 = acc_fun(tn + (h * __2), vn.clone() + k2.clone() * (h * __2));
    let k4 = acc_fun(tn + h, vn.clone() + (k3.clone() * h));
    let vn1 = vn.clone() + (k1 + k2 * _2 + k3 * _2 + k4) * (h * __6);

    let k1x = vn.clone();
    let k2x = vn.clone() + k1x.clone() * (h / _2);
    let k3x = vn.clone() + k2x.clone() * (h / _2);
    let k4x = vn + k3x.clone() * h;
    let xn1 = xn + (k1x + k2x * _2 + k3x * _2 + k4x) * (h * __6);

    return (xn1, vn1);
}

#[cfg(test)]
mod tests {
    extern crate nalgebra;
    extern crate test;
    use self::nalgebra::core::MatrixMN;
    use self::nalgebra::U1;
    use self::test::Bencher;
    use super::*;
    use std::ops::Index;

    #[test]
    fn generic() {
        pub type V = MatrixMN<f64, U1, U1>;
        let h = 0.01;

        let g = V::from([9.81]);
        let acc_fun = |_t, v| g - 1.0 * v;
        let mut t = Vec::new();
        let mut x = Vec::new();
        let mut v = Vec::new();

        let mut tn = 0.;
        let mut xn = V::zeros();
        let mut vn = V::zeros();

        for _ in 1..10000 {
            let a = rk4(tn, xn, vn, h, acc_fun);
            xn = a.0;
            vn = a.1;
            tn += h;
            t.push(tn);
            x.push(xn);
            v.push(vn);
        }
        assert_approx_eq!(
            xn.index((0, 0)),
            975.9142065967176,
            975.9142065967176 * 0.005
        )
    }
    #[test]
    fn expl() {
        let h = 0.01;

        let g = 9.81;
        let acc_fun = |_t, v| g - 1.0 * v;
        let mut t = Vec::new();
        let mut x = Vec::new();
        let mut v = Vec::new();

        let mut tn = 0.;
        let mut xn = 0.;
        let mut vn = 0.;

        for _ in 1..10000 {
            let a = rk4_one_var(tn, xn, vn, h, acc_fun);
            xn = a.0;
            vn = a.1;
            tn += h;
            t.push(tn);
            x.push(xn);
            v.push(vn);
        }
        assert_approx_eq!(xn, 975.9142065967176, 975.9142065967176 * 0.005)
    }

    #[bench]
    fn bench_expl(b: &mut Bencher) {
        let h = 0.01;
        let g = 9.81;
        let acc_fun = |_t, v| g - 1.0 * v;
        let mut tn = 0.;
        let mut xn = 0.;
        let mut vn = 0.;
        b.iter(|| {
            for _ in 0..10000 {
                let a = rk4_one_var(tn, xn, vn, h, acc_fun);
                xn = a.0;
                vn = a.1;
                tn += h;
            }
            test::black_box(xn);
        })
    }

    #[bench]
    fn bench_generic(b: &mut Bencher) {
        pub type V = MatrixMN<f64, U1, U1>;
        let h = 0.01;
        let g = V::from([9.81]);
        let acc_fun = |_t, v| g - 1.0 * v;

        let mut tn = 0.;
        let mut xn = V::zeros();
        let mut vn = V::zeros();

        b.iter(|| {
            for _ in 0..10000 {
                let a = rk4(tn, xn, vn, h, acc_fun);
                xn = a.0;
                vn = a.1;
                tn += h;
            }
            test::black_box(xn);
        });
    }
}
