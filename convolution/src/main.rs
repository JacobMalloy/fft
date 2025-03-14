use fft;
use imaginary;
use imaginary::imaginary::Imag;
use std::iter;

fn pad_2d<'a, T>(input: &'a [T], size: usize, size_new: usize) -> Box<[T]>
where
    T: From<isize>,
    T: Copy,
{
    let iter_gen = |x: &'a [T]| {
        x.iter()
            .copied()
            .chain(iter::repeat_n(0.into(), size_new - size))
    };
    let new_extend_iter = std::iter::repeat(0.into()).take((size_new - size) * size_new);
    input
        .chunks(size)
        .map(iter_gen)
        .flatten()
        .chain(new_extend_iter)
        .collect()
}

fn print_matrix<T: std::fmt::Display>(input: &[T], size: usize) {
    for chunk in input.chunks(size) {
        for num in chunk {
            print!("{num} ");
        }
        println!("");
    }
}

fn main() {
    let arr = [1, 0, -1, 1, 0, -1, 1, 0, -1];
    let new_arr = pad_2d(&arr, 3, 5);
    print_matrix(new_arr.as_ref(), 5);
    let fft1 = fft::fft2d(new_arr.as_ref(), 5);
    let arr2: Box<[isize]> = (1..=25).collect();
    print_matrix(arr2.as_ref(), 5);
    let fft2 = fft::fft2d(arr2.as_ref(), 5);
    let prod: Box<[Imag]> = fft1
        .0
        .iter()
        .zip(fft2.0.iter())
        .map(|(x, y)| *x * *y)
        .collect();
    let result = fft::ifft2d(prod.as_ref(), fft2.1);
    print_matrix(prod.as_ref(), result.1);
    println!("");
    print_matrix(result.0.as_ref(), result.1)
}
