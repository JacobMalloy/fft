use core::f64::consts::PI;
use imaginary::imaginary::Imag;
use std::iter;

pub fn dft<'a, T>(input: &'a [T]) -> Box<[Imag]>
where
    T: core::ops::Mul<Imag, Output = Imag> + Copy,
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

pub fn idft<'a, T>(input: &'a [T]) -> Box<[Imag]>
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

fn split_even_and_odd<T>(input: &mut [T], tmp_array: &mut [T])
where
    T: Copy,
{
    let n = input.len();

    debug_assert!(n % 2 == 0);

    debug_assert!(input.len() <= tmp_array.len());

    let tmp_its = tmp_array[0..n].split_at_mut(n / 2);

    let tmp = tmp_its.0.iter_mut().zip(tmp_its.1.iter_mut());

    let tmp2 = input.chunks_exact(2);

    for ((even_dst, odd_dst), chunk) in tmp.zip(tmp2) {
        *even_dst = chunk[0];
        *odd_dst = chunk[1];
    }
    input.copy_from_slice(&tmp_array[0..n]);
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

fn next_power2(n: usize) -> usize {
    let new_n = (1 as usize) << (usize::BITS - ((n - 1).leading_zeros()));
    debug_assert!(new_n >= n);
    debug_assert!((new_n >> 1) < n);
    debug_assert!(new_n.count_ones() == 1);
    new_n
}

fn expand_array_to_imag<T>(input: &[T]) -> Box<[Imag]>
where
    Imag: From<T>,
    T: Copy,
{
    let n = input.len();
    let new_n = next_power2(n);
    let extend_iter = std::iter::repeat(Imag::default()).take(new_n - n);
    input
        .iter()
        .copied()
        .map(|x| Imag::from(x))
        .chain(extend_iter)
        .collect()
}

fn fft_internal<const MUL_CONST: isize>(input: &mut [Imag]) {
    let return_vec = input;
    let new_n = return_vec.len();

    let mut factor: Box<[Imag]> = std::iter::repeat(Imag::default()).take(new_n / 2).collect();
    factor[0] = Imag {
        real: 1.0,
        imag: 0.0,
    };

    debug_assert!(return_vec.len().count_ones() == 1);

    let mut len = new_n;
    let mut tmp_array: Box<[Imag]> = iter::repeat(Imag::default()).take(len).collect();
    while len > 1 {
        for i in return_vec.chunks_exact_mut(len) {
            split_even_and_odd(i, &mut tmp_array);
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
}

pub fn fft<T>(input: &[T]) -> Box<[Imag]>
where
    Imag: From<T>,
    T: Copy,
{
    let mut return_value = expand_array_to_imag(input);
    fft_internal::<-2>(return_value.as_mut());
    return return_value;
}

pub fn ifft<T>(input: &[T]) -> Box<[Imag]>
where
    Imag: From<T>,
    T: Copy,
{
    let mut return_value = expand_array_to_imag(input);
    fft_internal::<2>(return_value.as_mut());
    let n = input.len();
    for i in return_value.iter_mut() {
        *i = (1.0 / (n as f64)) * *i;
    }
    return return_value;
}

//expands to square matrix where both dimension is power of 2
fn expand_2d_imag<'a, T>(input: &'a [T], width: usize) -> (Box<[Imag]>, usize)
where
    Imag: From<T>,
    T: Copy,
{
    let n = input.len();
    let height = n / width;
    let larger_size = width.max(height);
    let new_n = next_power2(larger_size);
    let extend_iter = std::iter::repeat(Imag::default()).take(new_n - width);
    let iter_gen = |x: &'a [T]| {
        x.iter()
            .copied()
            .map(|x| x.into())
            .chain(extend_iter.clone())
    };
    let new_extend_iter = std::iter::repeat(Imag::default()).take((new_n - height) * new_n);
    let tmp = input
        .chunks(width)
        .map(iter_gen)
        .flatten()
        .chain(new_extend_iter)
        .collect();
    (tmp, new_n)
}

fn square_transpose<T>(input: &mut [T], size: usize) {
    for x in 0..size {
        for y in x..size {
            input.swap(x * size + y, y * size + x)
        }
    }
}

pub fn fft2d<T>(input: &[T], width: usize) -> (Box<[Imag]>, usize)
where
    Imag: From<T>,
    T: Copy,
{
    let height = input.len() / width;
    let (mut return_vec, size) = expand_2d_imag(input, width);
    for w in return_vec.chunks_mut(size).take(height) {
        fft_internal::<-2>(w);
    }
    square_transpose(return_vec.as_mut(), size);
    for w in return_vec.chunks_mut(size) {
        fft_internal::<-2>(w);
    }
    square_transpose(return_vec.as_mut(), size);
    (return_vec, size)
}

pub fn ifft2d<T>(input: &[T], width: usize) -> (Box<[Imag]>, usize)
where
    Imag: From<T>,
    T: Copy,
{
    let height = input.len() / width;
    let (mut return_vec, size) = expand_2d_imag(input, width);
    let multiply_value: Imag = (1.0 / (size as f64)).into();
    for w in return_vec.chunks_mut(size).take(height) {
        fft_internal::<2>(w);
        w.iter_mut().for_each(|x| *x = *x * multiply_value);
    }
    square_transpose(return_vec.as_mut(), size);
    for w in return_vec.chunks_mut(size) {
        fft_internal::<2>(w);
        w.iter_mut().for_each(|x| *x = *x * multiply_value);
    }
    square_transpose(return_vec.as_mut(), size);
    (return_vec, size)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_even_and_odd_test() {
        let mut input = [0, 1, 2, 3, 4, 5, 6, 7];
        let mut tmp = [0, 0, 0, 0, 0, 0, 0, 0];
        let correct = [0, 2, 4, 6, 1, 3, 5, 7];
        split_even_and_odd(&mut input, &mut tmp);
        println!("{:?}", input);
        assert!(input.iter().zip(correct.iter()).all(|(x, y)| x == y));
    }

    #[test]
    fn square_transpose_test() {
        let mut arr = [1, 2, 3, 4, 5, 6, 7, 8, 9];

        square_transpose(&mut arr, 3);
        assert_eq!([1, 4, 7, 2, 5, 8, 3, 6, 9], arr);
    }

    #[test]
    fn fft2d_test() {
        let input = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let res = fft2d(&input, 4);
        let compare = std::iter::repeat_n(1, 16).map(|x| x.into()).collect();
        assert_eq!(res.0, compare)
    }

    #[test]
    fn ifft2d_test() {
        let correct: Box<[Imag]> = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            .iter()
            .map(|x| x.into())
            .collect();
        let compare = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
        let res = ifft2d(&compare, 4);
        assert_eq!(res.0, correct)
    }
}
