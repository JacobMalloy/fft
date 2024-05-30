use imaginary::imaginary::Imag;
use core::f64::consts::PI;

#[allow(dead_code)]
pub fn dft<'a, T>(input: &'a [T]) -> Vec<Imag>
where
    T: std::ops::Mul<Imag, Output = Imag> + Copy,
{
    let len = input.len();
    (0..len)
        .map(|k| {
            input
                .iter()
                .copied()
                .enumerate()
                .map(|(n, x)| x * Imag::euler(-2.0 * (n as f64) * PI * (k as f64) / (len as f64)))
                .sum::<Imag>()
        })
        .collect()
}

#[allow(dead_code)]
pub fn idft<'a, T>(input: &'a [T]) -> Vec<Imag>
where
    T: std::ops::Mul<Imag, Output = Imag> + Copy,
{
    let len = input.len();
    (0..len)
        .map(|k| {
            (1.0 / (len as f64))
                * input
                    .iter()
                    .copied()
                    .enumerate()
                    .map(|(n, x)| {
                        x * Imag::euler(2.0 * (n as f64) * PI * (k as f64) / (len as f64))
                    })
                    .sum::<Imag>()
        })
        .collect()
}

fn split_even_and_odd<T>(input: &mut [T]) 
where T:Copy
{
    let n = input.len();

    assert!(n%2==0);

    let even: Vec<T> = input
        .iter()
        .step_by(2)
        .copied()
        .collect::<Vec<T>>();
    let odd: Vec<T> = input
        .iter()
        .skip(1)
        .step_by(2)
        .copied()
        .collect::<Vec<T>>();

    input[0..n / 2].copy_from_slice(&even);
    input[n / 2..n].copy_from_slice(&odd);
}

fn combine_step(input: &mut [Imag], factor: &[Imag]) {
    let n = input.len();
    //let factor_mul = factor.len()/(n/2);
    let s = input.split_at_mut(n / 2);
    let even_iter = s.0.iter_mut();
    let odd_iter = s.1.iter_mut();
    let factor_iter = factor.iter().copied();
    for ((even, odd), factor) in even_iter.zip(odd_iter).zip(factor_iter) {
        let tmp = *odd * factor;
        (*even, *odd) = (*even + tmp, *even - tmp);
    }
}

fn fft_internal<const MUL_CONST: isize,T>(input: &[T]) -> Vec<Imag>
where
    Imag: From<T>,
    T: Copy,
{
    let n = input.len();
    let new_n: usize = (1 as usize) << (usize::BITS - ((n - 1).leading_zeros()));

    assert!(new_n >= n);
    assert!((new_n >> 1) < n);
    assert!(new_n.count_ones() == 1);

    let mut return_vec: Vec<Imag> = input.iter().copied().map(|x| Imag::from(x)).collect();
    return_vec.extend(
        std::iter::repeat(Imag {
            real: 0.0,
            imag: 0.0,
        })
        .take(new_n - n),
    );


    let mut factor: Vec<Imag> = std::iter::repeat(Imag {
        real: 1.0,
        imag: 0.0,
    })
    .take(new_n / 2)
    .collect();

    

    assert!(return_vec.len().count_ones() == 1);

    let mut len = new_n;
    while len > 1 {
        for i in return_vec.chunks_exact_mut(len) {
            split_even_and_odd(i);
        }
        len /= 2;
    }

    let mut len = 2;
    while len <= new_n {
        //println!("factors");
        for i in (0..len / 4).rev() {
            factor[i * 2] = factor[i];
            //println!("factor:{}",return_vec[i]);
        }
        let mutate_iterator = factor[0..len / 2].iter_mut().enumerate().skip(1);
        for (i, change) in mutate_iterator.step_by(2) {
            *change = Imag::euler((MUL_CONST as f64) * PI * ((i) as f64) / (len as f64));
        }
        for i in return_vec.chunks_exact_mut(len) {
            combine_step(i, &factor[0..len / 2])
        }
        len *= 2;
    }

    return return_vec;
}

pub fn fft<T>(input: &[T]) -> Vec<Imag>
where
    Imag: From<T>,
    T: Copy,
{ 
    let return_val = fft_internal::<-2,T>(input);
    return return_val;
}

pub fn ifft<T>(input: &[T]) -> Vec<Imag>
where
    Imag: From<T>,
    T: Copy,
{
    let mut return_vec = fft_internal::<2,T>(input);
    let n = input.len();
    for i in return_vec.iter_mut() {
        *i = (1.0 / (n as f64)) * *i;
    }
    return return_vec;
}





#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_even_and_odd_test() {
        let mut input = [0,1,2,3,4,5,6,7];
        let correct = [0,2,4,6,1,3,5,7];
        split_even_and_odd(&mut input);
        println!("{:?}",input);
        assert!(input.iter().zip(correct.iter()).all(|(x,y)|x==y));
    }
}

