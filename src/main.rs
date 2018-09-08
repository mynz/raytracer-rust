// raytrace-rust

#![allow(unused)]

extern crate nalgebra as na;

//type V3 = nalgebra::Vector3;

#[derive(Debug)]
struct Vec3 {
    x: f32,
    y: f32,
    z: f32,
    //v: [f32; 3]
}

fn main() {

    let v = na::Vector3::new(0, 1, 2);

    //let v = Vec3{0, 0, 0};

    println!("Hello, world!");
    
    println!("v: {:?}", v);
}
