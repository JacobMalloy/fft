use std::iter::Sum;
use std::ops::Add;
use std::ops::Mul;
use std::ops::Sub;

#[derive(Clone, Copy, PartialEq)]
pub struct Imag {
    pub real: f64,
    pub imag: f64,
}

impl Imag {
    pub fn euler(rad: f64) -> Imag {
        Imag {
            real: rad.cos(),
            imag: rad.sin(),
        }
    }
}

impl Default for Imag {
    fn default() -> Self {
        Imag {
            real: 0.0,
            imag: 0.0,
        }
    }
}

impl<T> Sum<T> for Imag
where
    Imag: Add<T, Output = Imag>,
{
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = T>,
    {
        let mut tmp = Imag {
            real: 0.0,
            imag: 0.0,
        };
        for i in iter {
            tmp = tmp + i;
        }
        return tmp;
    }
}

macro_rules! impl_op {
    ($lhs_type:ty,$rhs_type:ty,$return_type:ty,$trait:ident,$fn_name:ident,$block:expr) => {
        impl $trait<&$lhs_type> for &$rhs_type {
            type Output = $return_type;
            fn $fn_name(self, other: &$lhs_type) -> $return_type {
                return $block(self, other);
            }
        }
        impl $trait<$lhs_type> for $rhs_type {
            type Output = $return_type;
            fn $fn_name(self, other: $lhs_type) -> $return_type {
                return (&self).$fn_name(&other);
            }
        }
        impl $trait<&$lhs_type> for $rhs_type {
            type Output = $return_type;
            fn $fn_name(self, other: &$lhs_type) -> $return_type {
                return (&self).$fn_name(other);
            }
        }
        impl $trait<$lhs_type> for &$rhs_type {
            type Output = $return_type;
            fn $fn_name(self, other: $lhs_type) -> $return_type {
                return (self).$fn_name(&other);
            }
        }
    };
}

macro_rules! impl_from_for_number {
    ($($t:ty),*) => {
        $(
            impl From<$t> for Imag {
                fn from(v:$t)->Imag{
                    return Imag{real:v as f64,imag:0.0};
                }
            }
            impl From<&$t> for Imag {
                fn from(v:&$t)->Imag{
                    return Imag{real:*v as f64,imag:0.0};
                }
            }
        )*
    };
}
impl_from_for_number!(i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize, f32, f64);

macro_rules! impl_mul_for_number {
    ($($t:ty),*) => {
        $(
            impl_op!(Imag,$t,Imag,Mul,mul,|s:&$t,other:&Imag|other*Imag{real:*s as f64,imag:0.0});
        )*
    };
}
impl_mul_for_number!(i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize, f32, f64);

impl_op!(Imag, Imag, Imag, Add, add, |s: &Imag, other: &Imag| {
    Imag {
        real: s.real + other.real,
        imag: s.imag + other.imag,
    }
});
impl_op!(Imag, Imag, Imag, Sub, sub, |s: &Imag, other: &Imag| {
    Imag {
        real: s.real - other.real,
        imag: s.imag - other.imag,
    }
});
impl_op!(Imag, Imag, Imag, Mul, mul, |s: &Imag, other: &Imag| Imag {
    real: s.real * other.real - (s.imag * other.imag),
    imag: (s.real * other.imag) + (s.imag * other.real)
});

impl std::fmt::Display for Imag {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        if self.imag.abs() < 0.001 && self.real.abs() < 0.001 {
            write!(f, "0")
        } else if self.imag.abs() < 0.001 {
            write!(f, "{:.3}", self.real)
        } else if self.real.abs() < 0.001 {
            write!(f, "{:.3}i", self.imag)
        } else if self.imag < 0.0 {
            write!(f, "{:.3}{:.3}i", self.real, self.imag)
        } else {
            write!(f, "{:.3}+{:.3}i", self.real, self.imag)
        }
    }
}

impl std::fmt::Debug for Imag {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self)
    }
}
