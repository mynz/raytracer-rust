// raytrace-rust

#![allow(unused)]

extern crate image;
use std::io;
use std::io::prelude::*;

#[derive(Debug)]
struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub fn new(x: f32, y: f32, z: f32) -> Vec3 {
        Vec3 { x: x, y: y, z: z }
    }
    pub fn zero() -> Vec3 {
        Vec3 {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }
    pub fn dot(&self) -> f32 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }
    pub fn length(&self) -> f32 {
        self.dot().sqrt()
    }
    pub fn normailze(&self) -> Vec3 {
        let l = self.length();
        Vec3 {
            x: self.x / l,
            y: self.y / l,
            z: self.z / l,
        }
    }
}

#[test]
fn test_normalize() {
    let v = Vec3::new(1.0, 2.0, 3.0);
    let n = v.normailze();
    println!("v: {:?}, n: {:?}", v, n);
    assert!((1.0 - n.length()).abs() < 0.00001);
}

#[derive(Copy, Clone)]
struct Color(u8, u8, u8);

fn write_png() -> io::Result<()> {
    const NX: u32 = 100;
    const NY: u32 = 200;
    const NPIXELS: usize = (NX * NY) as usize;

    let mut buf = image::RgbImage::new(NX, NY);

    for (x, y, pixel) in buf.enumerate_pixels_mut() {
        let p = image::Rgb { data: [220, 0, 0] };
        *pixel = p;
    }

    image::save_buffer("sample.png", &buf, NX, NY, image::RGB(8))
}

fn main() {
    println!("Hello, world!");
    let v = Vec3::zero();
    println!("v: {:?}", v);

    write_png().unwrap()
}
