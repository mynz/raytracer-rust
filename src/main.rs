// raytrace-rust

#![allow(unused)]

extern crate image;
//use std::io;
use std::io::prelude::*;
use std::ops::{Add, Mul, Sub};

#[derive(Debug, Copy, Clone)]
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

    pub fn inv(&self) -> Vec3 {
        Vec3 {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }

    pub fn dot(&self, other: &Vec3) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }
    pub fn length(&self) -> f32 {
        self.dot(self).sqrt()
    }
    pub fn normailze(&self) -> Vec3 {
        let l = self.length();
        Vec3 {
            x: self.x / l,
            y: self.y / l,
            z: self.z / l,
        }
    }

    pub fn mul_f32(self, f: f32) -> Vec3 {
        Vec3 {
            x: self.x * f,
            y: self.y * f,
            z: self.z * f,
        }
    }
}

impl Add for Vec3 {
    type Output = Vec3;
    fn add(self, other: Vec3) -> Vec3 {
        Vec3 {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}

impl Sub for Vec3 {
    type Output = Vec3;
    fn sub(self, other: Vec3) -> Vec3 {
        Vec3 {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
}

impl Mul for Vec3 {
    type Output = Vec3;
    fn mul(self, other: Vec3) -> Vec3 {
        Vec3 {
            x: self.x * other.x,
            y: self.y * other.y,
            z: self.z * other.z,
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

struct Ray {
    pub a: Vec3,
    pub b: Vec3,
}

impl Ray {
    fn new(a: &Vec3, b: &Vec3) -> Ray {
        Ray { a: *a, b: *b }
    }

    fn origin(&self) -> Vec3 {
        self.a
    }
    fn direction(&self) -> Vec3 {
        self.b
    }
    fn point_at_parameter(&self, t: f32) -> Vec3 {
        self.b.mul_f32(t) + self.a
    }
}

fn hit_sphere(center: &Vec3, radius: f32, r: &Ray) -> f32 {
    let oc = r.origin() - *center;
    let a = r.direction().dot(&r.direction());
    let b = 2. * oc.dot(&r.direction());
    let c = oc.dot(&oc) - (radius * radius);
    let discriminant = b * b - 4. * a * c;
    if discriminant < 0. {
        -1.
    } else {
        (-b - discriminant.sqrt()) / (2. * a)
    }
}

fn color(r: &Ray) -> Vec3 {
    let t = hit_sphere(&Vec3::new(0., 0., -1.), 0.5, r);
    if t > 0. {
        let n = r.point_at_parameter(t) - Vec3::new(0., 0., -1.);
        return Vec3::new(n.x + 1.0, n.y + 1.0, n.z + 1.0).mul_f32(0.5);
    }

    let dir = r.direction().normailze();
    let t = 0.5 * (dir.y + 1.0);
    Vec3::new(1.0, 1.0, 1.0).mul_f32(1.0 - t) + (Vec3::new(0.5, 0.7, 1.0).mul_f32(t))
}

fn write_png() -> std::io::Result<()> {
    const NX: u32 = 200;
    const NY: u32 = 100;
    const NPIXELS: usize = (NX * NY) as usize;

    let lower_left_coner = Vec3::new(-2.0, -1.0, -1.0);
    let horizontal = Vec3::new(4., 0., 0.);
    let vertical = Vec3::new(0., 2., 0.);
    let origin = Vec3::new(0., 0., 0.);

    let mut buf = image::RgbImage::new(NX, NY);

    for (x, y, pixel) in buf.enumerate_pixels_mut() {
        let u = x as f32 / NX as f32;
        let v = 1.0 - y as f32 / NY as f32;

        let r = Ray::new(
            &origin,
            &(horizontal.mul_f32(u) + vertical.mul_f32(v) + lower_left_coner),
        );
        let col = color(&r);
        let rgb = image::Rgb {
            data: [
                (col.x * 255.99) as u8,
                (col.y * 255.99) as u8,
                (col.z * 255.99) as u8,
            ],
        };
        *pixel = rgb;
    }

    image::save_buffer("sample.png", &buf, NX, NY, image::RGB(8))
}

fn main() {
    println!("Hello, world!");
    let v = Vec3::zero();
    println!("v: {:?}", v);

    write_png().unwrap()
}
