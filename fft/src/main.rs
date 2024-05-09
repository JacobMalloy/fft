use imaginary::imaginary::Imag;
use std::{f64::consts::PI, hint::black_box, io::BufRead};

#[allow(dead_code)]
fn dft<'a, T>(input: &'a [T]) -> Vec<Imag>
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
fn idft<'a, T>(input: &'a [T]) -> Vec<Imag>
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

fn split_even_and_odd(input: &mut [Imag]) {
    let n = input.len();

    let even: Vec<Imag> = input.iter().step_by(2).map(|x| *x).collect::<Vec<Imag>>();
    let odd: Vec<Imag> = input
        .iter()
        .skip(1)
        .step_by(2)
        .map(|x| *x)
        .collect::<Vec<Imag>>();

    input[0..n / 2].copy_from_slice(&even);
    input[n / 2..n].copy_from_slice(&odd);
}

fn combine_step(input: &mut [Imag], factor: &[Imag]) {
    let n = input.len();
    //let factor_mul = factor.len()/(n/2);
    let s = input.split_at_mut(n / 2);
    let even_iter = s.0.iter_mut();
    let odd_iter = s.1.iter_mut();
    let factor_iter = factor.iter();
    for ((even, odd), factor) in even_iter.zip(odd_iter).zip(factor_iter) {
        let tmp = *odd * *factor;
        (*even, *odd) = (*even + tmp, *even - tmp);
    }
}

fn fft_internal<const MUL_CONST: isize>(input: &mut [Imag], factor: &mut [Imag]) {
    assert!(input.len().count_ones() == 1);
    let n = input.len();

    let mut len = n;
    while len > 1 {
        for i in input.chunks_exact_mut(len) {
            split_even_and_odd(i);
        }
        len /= 2;
    }

    let mut len = 2;
    while len <= n {
        //println!("factors");
        for i in (0..len / 4).rev() {
            factor[i * 2] = factor[i];
            //println!("factor:{}",input[i]);
        }
        let mutate_iterator = factor[0..len / 2].iter_mut().enumerate().skip(1);
        for (i, change) in mutate_iterator.step_by(2) {
            *change = Imag::euler((MUL_CONST as f64) * PI * ((i) as f64) / (len as f64));
        }
        for i in input.chunks_exact_mut(len) {
            combine_step(i, &factor[0..len / 2])
        }
        len *= 2;
    }
}

fn fft<T>(input: &[T]) -> Vec<Imag>
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
    //let factor:Vec<Imag> = (0..new_n/2).map(|i|Imag::euler(-2.0*PI*((i ) as f64)/(new_n as f64))).collect();
    let mut factor: Vec<Imag> = std::iter::repeat(Imag {
        real: 1.0,
        imag: 0.0,
    })
    .take(new_n / 2)
    .collect();
    fft_internal::<-2>(&mut return_vec, &mut factor);
    return return_vec;
}

fn ifft<T>(input: &[T]) -> Vec<Imag>
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
    //let factor:Vec<Imag> = (0..new_n/2).map(|i|Imag::euler(-2.0*PI*((i ) as f64)/(new_n as f64))).collect();
    let mut factor: Vec<Imag> = std::iter::repeat(Imag {
        real: 1.0,
        imag: 0.0,
    })
    .take(new_n / 2)
    .collect();
    fft_internal::<2>(&mut return_vec, &mut factor);
    for i in return_vec.iter_mut() {
        *i = (1.0 / (n as f64)) * *i;
    }
    return return_vec;
}

fn main() {
    let file = std::fs::File::open("../tmp.data").unwrap();
    let bufreader = std::io::BufReader::new(file);
    let data: Vec<i64> = bufreader
        .lines()
        .map(|x| x.unwrap().parse::<i64>().unwrap())
        .collect();

    let time = std::time::Instant::now();
    let _result = black_box(fft(&data));
    println!("{}", (time.elapsed().as_micros() as f64) / 1000000.0);

    let tmp = [
        -9.0, 4.0, -5.0, 7.0, -2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];
    let tmp2 = [
        1.0, 3.0, -5.0, 2.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];
    let a = fft(&tmp);
    let b = fft(&tmp2);
    /*for i in A.iter(){
       println!("{}",i);
    }
    println!("");
    for i in B.iter(){
       println!("{}",i);
    }

    println!("");*/
    let tmp3: Vec<Imag> = a.iter().zip(b.iter()).map(|(a, b)| a * b).collect();
    /*for i in tmp3.iter(){
        println!("{}",i)
    }
    */
    println!("");
    let tmp4: Vec<Imag> = ifft(&tmp3);
    for i in tmp4.iter() {
        println!("{}", i)
    }
}
