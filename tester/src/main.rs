use imaginary::imaginary::Imag;
use fft::{fft,ifft};
use std::{hint::black_box, io::BufRead};


fn main() {
    let path = std::env::args().nth(1).unwrap_or(String::from("../tmp.data"));
    let file = std::fs::File::open(path).unwrap();
    let bufreader = std::io::BufReader::new(file);
    let data: Box<[i64]> = bufreader
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
    let tmp3: Box<[Imag]> = a.iter().zip(b.iter()).map(|(a, b)| a * b).collect();
    /*for i in tmp3.iter(){
        println!("{}",i)
    }
    */
    println!("");
    let tmp4: Box<[Imag]> = ifft(&tmp3);
    for i in tmp4.iter() {
        println!("{}", i)
    }
}
