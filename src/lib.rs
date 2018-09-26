extern crate alga;
use alga::general::Real;
use alga::linear::VectorSpace;

// https://scicomp.stackexchange.com/questions/21060/runge-kutta-simulation-for-projectile-motion-with-drag
pub fn rk4<S, V>(tn: S, xn: V, vn: V, h: S, acc_fun: impl Fn(S, V) -> V) -> (V, V)
where
    S: Real,
    V: VectorSpace,
{
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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn it_works() {
        let h = 0.01;
        let acc_fun = |_t, v| 9.81 - 1.0 * v;
        let mut t = vec![0.];
        let mut x = vec![0.];
        let mut v = vec![0.];

        let mut tn = 0.;
        let mut xn = 0.;
        let mut vn = 0.;

        for _ in 1..10000 {
            let a = rk4(tn, xn, vn, h, acc_fun);
            xn = a.0;
            vn = a.1;
            tn += h;
            t.push(tn);
            x.push(xn);
            v.push(vn);
        }
    }
}
