use std::{f64::consts::PI, hint::black_box, io::BufRead};
use imaginary::imaginary::Imag;


fn dft<'a,T>(input:&'a[T])->Vec<Imag>
where T:std::ops::Mul<Imag,Output = Imag> + Copy{
    let N = input.len();
    (0..N).map(|k|input.iter().copied().enumerate().map(|(n,x)|x*Imag::euler(-2.0*(n as f64)*PI*(k as f64)/(N as f64))).sum::<Imag>()).collect()
}

fn idft<'a,T>(input:&'a[T])->Vec<Imag>
where T:std::ops::Mul<Imag,Output = Imag> + Copy{
    let N = input.len();
    (0..N).map(|k|(1.0/(N as f64))*input.iter().copied().enumerate().map(|(n,x)|x*Imag::euler(2.0*(n as f64)*PI*(k as f64)/(N as f64))).sum::<Imag>()).collect()
}


fn fft_internal(input:&mut [Imag],factor:&[Imag]){
    assert!(input.len().count_ones()==1);
    let N = input.len();
    if input.len() <= 1{
        //let tmp = dft(input);
        //input.copy_from_slice(&tmp);
        /*for (to, from) in input.iter_mut().zip(tmp.into_iter()){
            *to = from;
        }*/
        
        return;
    } 
    let even:Vec<Imag> = input.iter().step_by(2).map(|x|*x).collect::<Vec<Imag>>();
    let odd:Vec<Imag> = input.iter().skip(1).step_by(2).map(|x|*x).collect::<Vec<Imag>>();

    input[0..N/2].copy_from_slice(&even);
    input[N/2..N].copy_from_slice(&odd);
    drop(even);
    drop(odd);
   /*  
    for (to, from) in input[0..N/2].iter_mut().zip(even.into_iter()){
        *to = from;
    }
    for (to, from) in input[N/2..N].iter_mut().zip(odd.into_iter()){
        *to = from;
    }*/
    fft_internal(&mut input[0..N/2],factor);
    fft_internal(&mut input[N/2..N],factor);

    let factor_mul = factor.len()/N;
    //for i in 0..N/2{
    let s = input.split_at_mut(N/2);
    for ((even,odd),factor) in s.0.iter_mut().zip(s.1.iter_mut()).zip(factor.iter().step_by(factor_mul)){
        let tmp = *odd * *factor;
        (*even,*odd) = (*even+tmp ,*even-tmp );
        //println!("{} {}",input[i],input[i+(N/2)]);
    }
    
}



fn fft<T>(input:&[T])->Vec<Imag>
where Imag:From<T>,T:Copy{
    let N = input.len();
    let mut return_vec:Vec<Imag> = input.iter().copied().map(|x|Imag::from(x)).collect();
    let factor:Vec<Imag> = (0..N).map(|i|Imag::euler(-2.0*PI*((i ) as f64)/(N as f64))).collect();
    fft_internal(&mut return_vec,&factor);
    return return_vec;
}

/* 
fn fft<'a,T>(input:&'a[T])->Vec<Imag>
where T:std::ops::Mul<Imag,Output = Imag>, T:Copy
{
    assert!(input.len().count_ones()==1);
    let N = input.len();
    if input.len() <= 8{ 
        return dft(input);
    }
    
    let even:Vec<Imag> = fft(&input.iter().step_by(2).map(|x|*x).collect::<Vec<T>>());
    let odd:Vec<Imag> = fft(&input.iter().skip(1).step_by(2).map(|x|*x).collect::<Vec<T>>());
    let vals = (0..N/2).map(|n|Imag::euler(-2.0*PI*(n as f64)/(N as f64)));
    let vals2 = (N/2..N).map(|n|Imag::euler(-2.0*PI*(n as f64)/(N as f64)));
    let mut return_val:Vec<Imag> = vals.zip(even.iter().zip(odd.iter())).map(|(factor,(even,odd))|*even + (&factor*odd)).collect();
    return_val.extend(vals2.zip(even.iter().zip(odd.iter())).map(|(factor,(even,odd))|*even + (&factor*odd)));
    return return_val
}

*/


fn main() {
    let file = std::fs::File::open("../tmp2.data").unwrap();
    let bufreader = std::io::BufReader::new(file);
    let data:Vec<i64> = bufreader.lines().map(|x|x.unwrap().parse::<i64>().unwrap()).collect();
    
    let time = std::time::Instant::now(); 
    let result = black_box(fft(&data));
    println!("{}",(time.elapsed().as_micros() as f64)/1000000.0);
    
    
    let tmp = [-9.0,4.0,-5.0,7.0,-2.0,3.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0];
    let tmp2 = [1.0,3.0,-5.0,2.0,6.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0];
    let A = fft(&tmp);
    let B = fft(&tmp2);
    /*for i in A.iter(){
       println!("{}",i); 
    }
    println!("");
    for i in B.iter(){
       println!("{}",i); 
    }

    println!("");*/
    let tmp3:Vec<Imag> = A.iter().zip(B.iter()).map(|(a,b)|a * b).collect();
    /*for i in tmp3.iter(){
        println!("{}",i)
    }
    */
    println!("");
    let tmp4:Vec<Imag> = idft(&tmp3);
    for i in tmp4.iter(){
        println!("{}",i)
    }
    
}
