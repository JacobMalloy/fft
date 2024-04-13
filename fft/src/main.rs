
#![feature(portable_simd)]
use std::{f64::consts::PI, hint::black_box, io::BufRead};
use imaginary::imaginary::Imag;
use std::simd::ptr::SimdMutPtr;
use std::simd::Simd;
use std::simd::ptr::SimdConstPtr;

#[allow(dead_code)]
fn dft<'a,T>(input:&'a[T])->Vec<Imag>
where T:std::ops::Mul<Imag,Output = Imag> + Copy{
    let len = input.len();
    (0..len).map(|k|input.iter().copied().enumerate().map(|(n,x)|x*Imag::euler(-2.0*(n as f64)*PI*(k as f64)/(len as f64))).sum::<Imag>()).collect()
}

fn idft<'a,T>(input:&'a[T])->Vec<Imag>
where T:std::ops::Mul<Imag,Output = Imag> + Copy{
    let len = input.len();
    (0..len).map(|k|(1.0/(len as f64))*input.iter().copied().enumerate().map(|(n,x)|x*Imag::euler(2.0*(n as f64)*PI*(k as f64)/(len as f64))).sum::<Imag>()).collect()
}

fn split_even_and_odd(input:&mut [Imag]){
    let n = input.len();

    let even:Vec<Imag> = input.iter().step_by(2).map(|x|*x).collect::<Vec<Imag>>();
    let odd:Vec<Imag> = input.iter().skip(1).step_by(2).map(|x|*x).collect::<Vec<Imag>>();

    input[0..n/2].copy_from_slice(&even);
    input[n/2..n].copy_from_slice(&odd);

}
#[inline(never)]
fn combine_step<const vectorize:bool>(input:&mut [Imag],factor:&[Imag]){
    let n = input.len();
    let blocks = if vectorize{n/8}else{0};
    let mut even_real_addrs = [0 as *mut f64,0 as *mut f64,0 as *mut f64,0 as *mut f64];
    let stride_length=if n > 1{
            let ptr1: *const f64 = &input[0].imag;
            let ptr2: *const f64 = &input[1].imag;
            (ptr2 as usize) - (ptr1 as usize)
        }else {
            0
        };
    let strides_array = [0,stride_length,2*stride_length,3*stride_length];
    let stride_vector = std::simd::Simd::from_array(strides_array);
    

    //let factor_mul = factor.len()/(n/2);
    
    let (even_slice,odd_slice) = input.split_at_mut(n/2);

    let mut even_real_addrs = Simd::splat(((&mut even_slice[0].real) as *mut f64) as usize) + stride_vector; 
    let mut even_imag_addrs = Simd::splat(((&mut even_slice[0].imag) as *mut f64) as usize) + stride_vector; 
    let mut odd_real_addrs = Simd::splat(((&mut odd_slice[0].real) as *mut f64) as usize) + stride_vector; 
    let mut odd_imag_addrs = Simd::splat(((&mut odd_slice[0].imag) as *mut f64) as usize) + stride_vector; 
    let mut factor_real_addrs = Simd::splat(((&mut factor[0].real) as *mut f64) as usize) + stride_vector; 
    let mut factor_imag_addrs = Simd::splat(((&mut factor[0].imag) as *mut f64) as usize) + stride_vector; 
    

    let stride_vector:std::simd::Simd<usize,4> = std::simd::Simd::splat(4*stride_length);

    for block in 0..blocks{
        let simd_even_real = <Simd<*const f64,4> as SimdConstPtr>::from_exposed_addr(even_real_addrs);
        let simd_even_imag = <Simd<*const f64,4> as SimdConstPtr>::from_exposed_addr(even_imag_addrs);
        let simd_odd_real = <Simd<*const f64,4> as SimdConstPtr>::from_exposed_addr(odd_real_addrs);
        let simd_odd_imag = <Simd<*const f64,4> as SimdConstPtr>::from_exposed_addr(odd_imag_addrs);
        let simd_factor_real = <Simd<*const f64,4> as SimdConstPtr>::from_exposed_addr(factor_real_addrs);
        let simd_factor_imag = <Simd<*const f64,4> as SimdConstPtr>::from_exposed_addr(factor_imag_addrs);
        
        
        let even_real = unsafe{std::simd::f64x4::gather_ptr(simd_even_real )};
        let even_imag = unsafe{std::simd::f64x4::gather_ptr(simd_even_imag )};  

        let odd_real = unsafe{std::simd::f64x4::gather_ptr(simd_odd_real )};
        let odd_imag = unsafe{std::simd::f64x4::gather_ptr(simd_odd_imag )}; 
        //println!("{:?}",odd_real);
        
        let factor_real = unsafe{std::simd::f64x4::gather_ptr(simd_factor_real )};
        let factor_imag = unsafe{std::simd::f64x4::gather_ptr(simd_factor_imag )};
        

        let tmp_real = (odd_real * factor_real) - (odd_imag * factor_imag);
        let tmp_imag = (odd_real * factor_imag) + (odd_imag * factor_real);

        //println!("tmp imag {:?}",tmp_imag);
        let tmp_even_real = even_real + tmp_real;
        let tmp_even_imag = even_imag + tmp_imag;

        let tmp_odd_real = even_real - tmp_real;
        let tmp_odd_imag = even_imag - tmp_imag;



        let simd_even_real = <Simd<*mut f64,4> as SimdMutPtr>::from_exposed_addr(even_real_addrs);
        let simd_even_imag = <Simd<*mut f64,4> as SimdMutPtr>::from_exposed_addr(even_imag_addrs);
        let simd_odd_real = <Simd<*mut f64,4> as SimdMutPtr>::from_exposed_addr(odd_real_addrs);
        let simd_odd_imag = <Simd<*mut f64,4> as SimdMutPtr>::from_exposed_addr(odd_imag_addrs);
 

        unsafe{tmp_even_real.scatter_ptr(simd_even_real)};
        unsafe{tmp_even_imag.scatter_ptr(simd_even_imag)};


        unsafe{tmp_odd_real.scatter_ptr(simd_odd_real)};
        unsafe{tmp_odd_imag.scatter_ptr(simd_odd_imag)};

        even_real_addrs += stride_vector;
        even_imag_addrs += stride_vector;
        odd_real_addrs += stride_vector;
        odd_imag_addrs += stride_vector;
        factor_real_addrs += stride_vector;
        factor_imag_addrs += stride_vector;
    }


    let s = input.split_at_mut(n/2);
    let even_iter = s.0.iter_mut(); 
    let odd_iter = s.1.iter_mut();
    let factor_iter = factor.iter();
    let tmp_iter =  even_iter.zip(odd_iter).zip(factor_iter);
    for ((even,odd),factor) in tmp_iter.skip(blocks*4){
        let tmp = *odd * *factor;
        (*even,*odd) = (*even+tmp ,*even-tmp );
    }
}


fn fft_internal(input:&mut [Imag],factor:&mut [Imag]){
    assert!(input.len().count_ones()==1);
    let n = input.len();

    let mut len = n;
    while len > 1{
        for i in input.chunks_exact_mut(len){
            split_even_and_odd(i);
        }
        len /= 2;
    }

    let mut len = 2;
    while len <= n{
        //println!("factors");
        for i in (0..len/4).rev(){
            factor[i*2] = factor[i];
            //println!("factor:{}",input[i]);
        }
        let mutate_iterator = factor[0..len/2].iter_mut().enumerate().skip(1);
        for (i,change) in mutate_iterator.step_by(2){
            let tmp = -2.0*PI/(len as f64); 
            *change = Imag::euler((i as f64)*tmp);
        }
        for i in input.chunks_exact_mut(len){
            combine_step::<true>(i, &factor[0..len/2])
        }
        len *= 2;
    }  
}



fn fft<T>(input:&[T])->Vec<Imag>
where Imag:From<T>,T:Copy{
    let n = input.len();
    let new_n:usize = (1 as usize)<<(usize::BITS-((n-1).leading_zeros()));
    
    assert!(new_n>=n);
    assert!((new_n>>1) < n);
    assert!(new_n.count_ones()==1);

    let mut return_vec:Vec<Imag> = input.iter().copied().map(|x|Imag::from(x)).collect();
    return_vec.extend(std::iter::repeat(Imag{real:0.0,imag:0.0}).take(new_n-n));
    //let factor:Vec<Imag> = (0..new_n/2).map(|i|Imag::euler(-2.0*PI*((i ) as f64)/(new_n as f64))).collect();
    let mut factor:Vec<Imag> = std::iter::repeat(Imag{real:1.0,imag:0.0}).take(new_n/2).collect();
    fft_internal(&mut return_vec,&mut factor);
    return return_vec;
}

fn main() { 
    /* 
    let mut tmp = [Imag{real:1.0,imag:1.0},
                          Imag{real:2.0,imag:2.0},
                          Imag{real:3.0,imag:3.0},
                          Imag{real:4.0,imag:4.0},
                          Imag{real:5.0,imag:5.0},
                          Imag{real:6.0,imag:6.0},
                          Imag{real:7.0,imag:7.0},
                          Imag{real:8.0,imag:8.0},
                          Imag{real:9.0,imag:9.0},
                          Imag{real:10.0,imag:10.0},
                          Imag{real:11.0,imag:11.0},
                          Imag{real:12.0,imag:12.0},
                          Imag{real:13.0,imag:13.0},
                          Imag{real:14.0,imag:14.0},
                          Imag{real:15.0,imag:15.0},
                          Imag{real:16.0,imag:16.0},

    ];
    let mut tmp3 = [Imag{real:1.0,imag:1.0},
                          Imag{real:2.0,imag:2.0},
                          Imag{real:3.0,imag:3.0},
                          Imag{real:4.0,imag:4.0},
                          Imag{real:5.0,imag:5.0},
                          Imag{real:6.0,imag:6.0},
                          Imag{real:7.0,imag:7.0},
                          Imag{real:8.0,imag:8.0},
                          Imag{real:9.0,imag:9.0},
                          Imag{real:10.0,imag:10.0},
                          Imag{real:11.0,imag:11.0},
                          Imag{real:12.0,imag:12.0},
                          Imag{real:13.0,imag:13.0},
                          Imag{real:14.0,imag:14.0},
                          Imag{real:15.0,imag:15.0},
                          Imag{real:16.0,imag:16.0},

    ]; 
    let tmp2 = [Imag{real:1.0,imag:0.0},
                          Imag{real:2.0,imag:0.0},
                          Imag{real:3.0,imag:0.0},
                          Imag{real:4.0,imag:0.0},
                          Imag{real:5.0,imag:0.0},
                          Imag{real:6.0,imag:0.0},
                          Imag{real:7.0,imag:0.0},
                          Imag{real:8.0,imag:0.0},
    ];
    combine_step::<true>(&mut tmp, &tmp2);
    combine_step::<false>(&mut tmp3, &tmp2);
    for (t,f) in tmp.iter().zip(tmp3.iter()){
        println!("{t} {f}")
    }
    return;
   */ 
    
    /*let file = std::fs::File::open("../tmp.data").unwrap();
    let bufreader = std::io::BufReader::new(file);
    let data:Vec<i64> = bufreader.lines().map(|x|x.unwrap().parse::<i64>().unwrap()).collect();
    
    let time = std::time::Instant::now(); 
    let _result = black_box(fft(&data));
    println!("{}",(time.elapsed().as_micros() as f64)/1000000.0);
   */ 
    
    let tmp = [-9.0,4.0,-5.0,7.0,-2.0,3.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0];
    let tmp2 = [1.0,3.0,-5.0,2.0,6.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0];
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
    let tmp3:Vec<Imag> = a.iter().zip(b.iter()).map(|(a,b)|a * b).collect();
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
