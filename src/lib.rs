extern crate alga;
use alga::general::Real;
use alga::linear::VectorSpace;

// https://scicomp.stackexchange.com/questions/21060/runge-kutta-simulation-for-projectile-motion-with-drag
pub fn rk4<S, V>(tn: S, xn: V, vn: V, h: S, acc_fun: impl Fn(S, V) -> V) -> (V, V)
where
    S: Real,
    V: VectorSpace<Field = S> + Copy,
{
    let _2 = S::from_i32(2).unwrap();
    let _6 = S::from_i32(6).unwrap();

    let __2 = _2.recip();
    let __6 = _6.recip();

    let k1 = acc_fun(tn, vn);
    let k2 = acc_fun(tn + (h * __2), vn + k1 * (h * __2));
    let k3 = acc_fun(tn + (h * __2), vn + k2 * (h * __2));
    let k4 = acc_fun(tn + h, vn + (k3 * h));
    let vn1 = vn + (k1 + k2 * _2 + k3 * _2 + k4) * (h * __6);

    let k1x = vn;
    let k2x = vn + k1x * (h / _2);
    let k3x = vn + k2x * (h / _2);
    let k4x = vn + k3x * h;
    let xn1 = xn + (k1x + k2x * _2 + k3x * _2 + k4x) * (h * __6);

    return (xn1, vn1);
}

#[cfg(test)]
mod tests {
    extern crate nalgebra;
    use self::nalgebra::core::MatrixMN;
    use self::nalgebra::U1;
    use super::*;

    #[test]
    fn it_works() {
        pub type V = MatrixMN<f64, U1, U1>;
        let h = 0.01;

        let g = V::from([9.81]);
        let acc_fun = |_t, v| g - 1.0 * v;
        let mut t = Vec::new();
        let mut x = Vec::new();
        let mut v = Vec::new();

        let mut tn = 0.;
        let mut xn = V::zeros_generic(U1, U1);
        let mut vn = V::zeros_generic(U1, U1);

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
