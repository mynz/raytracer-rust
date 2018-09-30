// raytrace-rust

#![allow(unused)]

extern crate image;
//use std::io;
use std::f32;
use std::io::prelude::*;
use std::ops::{Add, Div, Mul, Neg, Sub};

extern crate rand;
//use rand;

#[derive(Debug, Copy, Clone, PartialEq)]
struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub fn new(x: f32, y: f32, z: f32) -> Vec3 {
        Vec3 { x, y, z }
    }
    pub fn zero() -> Vec3 {
        Vec3 {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }
    pub fn one() -> Vec3 {
        Vec3::new(1., 1., 1.)
    }
    pub fn dot(&self, other: &Vec3) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn cross(v1: &Vec3, v2: &Vec3) -> Vec3 {
        let x = v1.y * v2.z - v1.z * v2.y;
        let y = -(v1.x * v2.z - v1.z * v2.x);
        let z = v1.x * v2.y - v1.y * v2.x;
        Vec3 { x, y, z }
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
}

impl Neg for Vec3 {
    type Output = Vec3;
    fn neg(self) -> Vec3 {
        Vec3 {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

trait VecOps {
    fn add_vec3(lhs: Vec3, rhs: Self) -> Vec3;
    fn sub_vec3(lhs: Vec3, rhs: Self) -> Vec3;
    fn mul_vec3(lhs: Vec3, rhs: Self) -> Vec3;
    fn div_vec3(lhs: Vec3, rhs: Self) -> Vec3;
}

impl VecOps for f32 {
    fn add_vec3(lhs: Vec3, rhs: Self) -> Vec3 {
        Vec3::new(lhs.x + rhs, lhs.y + rhs, lhs.z + rhs)
    }
    fn sub_vec3(lhs: Vec3, rhs: Self) -> Vec3 {
        Vec3::new(lhs.x - rhs, lhs.y - rhs, lhs.z - rhs)
    }
    fn mul_vec3(lhs: Vec3, rhs: Self) -> Vec3 {
        Vec3::new(lhs.x * rhs, lhs.y * rhs, lhs.z * rhs)
    }
    fn div_vec3(lhs: Vec3, rhs: Self) -> Vec3 {
        Vec3::new(lhs.x / rhs, lhs.y / rhs, lhs.z / rhs)
    }
}

impl VecOps for Vec3 {
    fn add_vec3(lhs: Vec3, rhs: Self) -> Vec3 {
        Vec3::new(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z)
    }
    fn sub_vec3(lhs: Vec3, rhs: Self) -> Vec3 {
        Vec3::new(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z)
    }
    fn mul_vec3(lhs: Vec3, rhs: Self) -> Vec3 {
        Vec3::new(lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z)
    }
    fn div_vec3(lhs: Vec3, rhs: Self) -> Vec3 {
        Vec3::new(lhs.x / rhs.x, lhs.y / rhs.y, lhs.z / rhs.z)
    }
}

impl<T: VecOps> Add<T> for Vec3 {
    type Output = Vec3;
    fn add(self, rhs: T) -> Vec3 {
        T::add_vec3(self, rhs)
    }
}
impl<T: VecOps> Sub<T> for Vec3 {
    type Output = Vec3;
    fn sub(self, rhs: T) -> Vec3 {
        T::sub_vec3(self, rhs)
    }
}
impl<T: VecOps> Mul<T> for Vec3 {
    type Output = Vec3;
    fn mul(self, rhs: T) -> Vec3 {
        T::mul_vec3(self, rhs)
    }
}
impl<T: VecOps> Div<T> for Vec3 {
    type Output = Vec3;
    fn div(self, rhs: T) -> Vec3 {
        T::div_vec3(self, rhs)
    }
}

#[test]
fn test_mul() {
    assert_eq!(Vec3::new(1., 0., 0.) * 2.0, Vec3::new(2.0, 0., 0.));
    assert_eq!(
        Vec3::new(1., 0., 0.) * Vec3::new(2.0, 0., 0.),
        Vec3::new(2.0, 0., 0.)
    );
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
        self.b * t + self.a
    }
}

trait Material {
    fn scatter(&self, r: &Ray, rec: &HitRecord) -> Option<(Ray, Vec3)>;
}

struct Lambertian {
    pub albedo: Vec3,
}

impl Material for Lambertian {
    fn scatter(&self, r: &Ray, rec: &HitRecord) -> Option<(Ray, Vec3)> {
        let target = rec.p + rec.normal + random_in_unit_sphere();
        Some((Ray::new(&rec.p, &(target - rec.p)), self.albedo))
    }
}

struct Metal {
    albedo: Vec3,
    fuzz: f32,
}

fn reflect(v: &Vec3, n: &Vec3) -> Vec3 {
    *v - (*n * Vec3::dot(v, n)) * 2.
}

impl Material for Metal {
    fn scatter(&self, r: &Ray, rec: &HitRecord) -> Option<(Ray, Vec3)> {
        let reflected = reflect(&Vec3::normailze(&r.direction()), &rec.normal);
        let b = reflected + random_in_unit_sphere() * self.fuzz;
        let scattered = Ray::new(&rec.p, &b);
        if Vec3::dot(&scattered.direction(), &rec.normal) > 0.0 {
            Some((scattered, self.albedo))
        } else {
            None
        }
    }
}

struct Dielectric {
    pub ref_idx: f32,
}

fn refract(v: &Vec3, n: &Vec3, ni_over_nt: f32) -> Option<Vec3> {
    let uv = v.normailze();
    let dt = uv.dot(n);
    let discriminant = 1.0 - ni_over_nt * ni_over_nt * (1. - dt * dt);
    if discriminant > 0.0 {
        Some((uv - *n * dt) * ni_over_nt - *n * discriminant.sqrt())
    } else {
        None
    }
}

fn schlick(cosine: f32, ref_idx: f32) -> f32 {
    let mut r0 = (1.0 - ref_idx) / (1.0 + ref_idx);
    r0 = r0 * r0;
    r0 + (1.0 - r0) * f32::powf((1.0 - cosine), 5.0)
}

impl Material for Dielectric {
    fn scatter(&self, r: &Ray, rec: &HitRecord) -> Option<(Ray, Vec3)> {
        let outward_normal;
        let reflected = reflect(&r.direction(), &rec.normal);
        let ni_over_nt;
        let attenuation = Vec3::new(1., 1., 1.);

        let reflect_prob;
        let cosine;

        if Vec3::dot(&r.direction(), &rec.normal) > 0.0 {
            outward_normal = -rec.normal;
            ni_over_nt = self.ref_idx;
            cosine = self.ref_idx * Vec3::dot(&r.direction(), &rec.normal) / r.direction().length();
        } else {
            outward_normal = rec.normal;
            ni_over_nt = 1.0 / self.ref_idx;
            cosine = -Vec3::dot(&r.direction(), &rec.normal) / r.direction().length();
        }

        let scattered;
        let refracted;
        if let Some(refracted_0) = refract(&r.direction(), &outward_normal, ni_over_nt) {
            reflect_prob = schlick(cosine, self.ref_idx);
            refracted = refracted_0
        } else {
            //scattered = Ray::new(&rec.p, &reflected);
            reflect_prob = 1.0;
            refracted = Vec3::zero();
        }

        if drand48() < reflect_prob {
            scattered = Ray::new(&rec.p, &reflected);
        } else {
            scattered = Ray::new(&rec.p, &refracted);
        }

        Some((scattered, attenuation))
    }
}

//#[derive(Debug, Copy, Clone)]
struct HitRecord<'a> {
    pub t: f32,
    pub p: Vec3,
    pub normal: Vec3,
    pub mat: &'a Material,
}

trait Hitable {
    fn hit(&self, r: &Ray, t_min_max: (f32, f32)) -> Option<HitRecord>;
}

struct Sphere<T: Material> {
    pub center: Vec3,
    pub radius: f32,
    pub mat: T,
}

impl<T: Material> Sphere<T> {
    fn new(center: &Vec3, radius: f32, mat: T) -> Sphere<T> {
        Sphere {
            center: *center,
            radius: radius,
            mat: mat,
        }
    }
}

impl<T: Material> Hitable for Sphere<T> {
    fn hit(&self, r: &Ray, t_min_max: (f32, f32)) -> Option<HitRecord> {
        let oc = r.origin() - self.center;
        let a = Vec3::dot(&r.direction(), &r.direction());
        let b = Vec3::dot(&oc, &r.direction());
        let c = Vec3::dot(&oc, &oc) - self.radius * self.radius;
        let discriminant = b * b - a * c;
        if discriminant > 0. {
            let temp = (-b - (b * b - a * c).sqrt()) / a;
            if temp < t_min_max.1 && temp > t_min_max.0 {
                let p = r.point_at_parameter(temp);
                return Some(HitRecord {
                    t: temp,
                    p: p,
                    normal: (p - self.center) / self.radius,
                    mat: &self.mat,
                });
            }
            let temp = (-b + (b * b - a * c).sqrt()) / a;
            if temp < t_min_max.1 && temp > t_min_max.0 {
                let p = r.point_at_parameter(temp);
                return Some(HitRecord {
                    t: temp,
                    p: p,
                    normal: (p - self.center) / self.radius,
                    mat: &self.mat,
                });
            }
        }
        None
    }
}

struct HitableList<'a> {
    pub list: Vec<Box<Hitable + 'a>>,
}

impl<'a> HitableList<'a> {
    fn new() -> HitableList<'a> {
        HitableList { list: Vec::new() }
    }
    fn with_capacity(n: usize) -> HitableList<'a> {
        HitableList {
            list: Vec::with_capacity(n),
        }
    }

    fn add(&mut self, item: Box<Hitable + 'a>) {
        self.list.push(item);
    }
}

impl<'a> Hitable for HitableList<'a> {
    fn hit(&self, r: &Ray, t_min_max: (f32, f32)) -> Option<HitRecord> {
        let mut temp_re: Option<HitRecord> = None;
        let mut closest_so_far = t_min_max.1;
        for v in self.list.iter() {
            if let Some(h) = v.hit(r, (t_min_max.0, closest_so_far)) {
                closest_so_far = h.t;
                temp_re = Some(h);
            }
        }
        temp_re
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

struct Camera {
    origin: Vec3,
    lower_left_corner: Vec3,
    horizontal: Vec3,
    vertical: Vec3,
    u: Vec3,
    v: Vec3,
    w: Vec3,
    lens_radius: f32,
}

impl Camera {
    fn new(
        lookfrom: &Vec3,
        lookat: &Vec3,
        vup: &Vec3,
        vfov: f32,
        aspect: f32,
        aperture: f32,
        focus_dist: f32,
    ) -> Camera {
        let lens_radius = aperture / 2.0;
        let theta = vfov * f32::consts::PI / 180.0;
        let half_height = (theta / 2.).tan();
        let half_width = aspect * half_height;

        let origin = *lookfrom;
        let w = Vec3::normailze(&(*lookfrom - *lookat));
        let u = Vec3::normailze(&Vec3::cross(vup, &w));
        let v = Vec3::cross(&w, &u);

        //let lower_left_corner = origin - u * half_width - v * half_height - w;
        let lower_left_corner =
            origin - u * focus_dist * half_width - v * focus_dist * half_height - w * focus_dist;
        let horizontal = u * half_width * focus_dist * 2.0;
        let vertical = v * half_height * focus_dist * 2.0;

        Camera {
            lower_left_corner,
            horizontal,
            vertical,
            origin,
            u,
            v,
            w,
            lens_radius,
        }
    }

    fn get_ray(&self, u: f32, v: f32) -> Ray {
        let rd = random_in_unit_disk() * self.lens_radius;
        let offset = self.u * rd.x + v * rd.y;
        let b =
            self.lower_left_corner + self.horizontal * u + self.vertical * v - self.origin - offset;
        Ray::new(&(self.origin + offset), &b)
    }
}

fn random_in_unit_sphere() -> Vec3 {
    let mut p = Vec3::zero();
    loop {
        p = Vec3::new(drand48(), drand48(), drand48()) * 2.0 - Vec3::one();
        if !(p.dot(&p) >= 1.0) {
            break;
        }
    }
    p
}

fn random_in_unit_disk() -> Vec3 {
    let mut p;
    loop {
        p = Vec3::new(drand48(), drand48(), 0.0) * 2.0 - Vec3::new(1., 1., 0.);
        if Vec3::dot(&p, &p) < 1.0 {
            break p;
        }
    }
}

fn color(r: &Ray, world: &Hitable, depth: u32) -> Vec3 {
    if let Some(rec) = world.hit(r, (0.001, f32::MAX)) {
        match rec.mat.scatter(r, &rec) {
            Some((ref scattered, attenuation)) if depth < 50 => {
                color(&scattered, world, depth + 1) * attenuation
            }
            _ => Vec3::zero(),
        }
    } else {
        let dir = r.direction().normailze();
        let t = 0.5 * (dir.y + 1.);
        Vec3::new(1., 1., 1.) * (1.0 - t) + Vec3::new(0.5, 0.7, 1.0) * t
    }
}

fn drand48() -> f32 {
    rand::random::<f32>()
}

fn random_scene<'a>() -> HitableList<'a> {
    const N: usize = 500;
    let mut list = HitableList::with_capacity(N);

    let mat = Lambertian {
        albedo: Vec3::new(0.5, 0.5, 0.5),
    };
    list.add(Box::new(Sphere::new(
        &Vec3::new(0., -1000., 0.),
        1000.,
        mat,
    )));

    for a in -11..11 {
        for b in -11..11 {
            let choose_mat = drand48();
            let center = Vec3::new(a as f32 + 0.9 * drand48(), 0.2, b as f32 + 0.9 * drand48());
            let s: Box<Hitable> = match choose_mat {
                d if d < 0.8 => {
                    let albedo = Vec3::new(
                        drand48() * drand48(),
                        drand48() * drand48(),
                        drand48() * drand48(),
                    );
                    Box::new(Sphere::new(&center, 0.2, Lambertian { albedo }))
                }
                d if d < 0.95 => {
                    let albedo = Vec3::new(
                        0.5 * (1.0 + drand48()),
                        0.5 * (1.0 + drand48()),
                        0.5 * (1.0 + drand48()),
                    );
                    let mat = Metal {
                        albedo,
                        fuzz: 0.5 * drand48(),
                    };
                    Box::new(Sphere::new(&center, 0.2, mat))
                }
                _ => {
                    let mat = Dielectric { ref_idx: 1.5 };
                    Box::new(Sphere::new(&center, 0.2, mat))
                }
            };
            list.add(s);
        }
    }
    let mat = Dielectric { ref_idx: 1.5 };
    list.add(Box::new(Sphere::new(&Vec3::new(0.0, 1.0, 0.0), 1.0, mat)));
    let mat = Lambertian {
        albedo: Vec3::new(0.4, 0.2, 0.1),
    };
    list.add(Box::new(Sphere::new(&Vec3::new(-4.0, 1.0, 0.0), 1.0, mat)));
    let mat = Metal {
        albedo: Vec3::new(0.7, 0.6, 0.5),
        fuzz: 0.0,
    };
    list.add(Box::new(Sphere::new(&Vec3::new(4.0, 1.0, 0.0), 1.0, mat)));
    list
}

fn simple_scene<'a>() -> HitableList<'a> {
    let mut world = HitableList::new();
    world.add(Box::new(Sphere::new(
    &Vec3::new(0., 0., -1.),
    0.5,
    Lambertian {
    albedo: Vec3::new(0.8, 0.3, 0.3),
    },
    )));
    world.add(Box::new(Sphere::new(
    &Vec3::new(0., -100.5, -1.0),
    100.0,
    Lambertian {
    albedo: Vec3::new(0.8, 0.8, 0.0),
    },
    )));
    world.add(Box::new(Sphere::new(
    &Vec3::new(1., 0., -1.),
    0.5,
    Metal {
    albedo: Vec3::new(0.8, 0.6, 0.2),
    fuzz: 0.3,
    },
    )));
    world.add(Box::new(Sphere::new(
    &Vec3::new(-1., 0., -1.),
    0.5,
    Dielectric { ref_idx: 1.5 },
    )));
    world.add(Box::new(Sphere::new(
    &Vec3::new(-1., 0., -1.),
    -0.45,
    Dielectric { ref_idx: 1.5 },
    )));

    world
}

fn write_png() -> std::io::Result<()> {
    const NX: u32 = 200;
    const NY: u32 = 150;
    const NS: u32 = 10;
    const NPIXELS: usize = (NX * NY) as usize;

    let world = random_scene();
    //let world = simple_scene();

    let lookfrom = Vec3::new(8.0, 1.5, 3.0);
    let lookat = Vec3::new(0.0, 1.0, 0.0);
    let dist_to_focus = (lookfrom - lookat).length();
    let aperture = 0.1;
    let cam = Camera::new(
        &lookfrom,
        &lookat,
        &Vec3::new(0.0, 1.0, 0.0),
        45.0,
        NX as f32 / NY as f32,
        aperture,
        dist_to_focus,
    );

    let mut buf = image::RgbImage::new(NX, NY);
    for (x, y, pixel) in buf.enumerate_pixels_mut() {
        let mut col = Vec3::zero();
        for s in 0..NS {
            let u = (x as f32 + drand48()) / NX as f32;
            //let v = (y as f32 + drand48()) / NY as f32;
            let v = 1.0 - ((y as f32 + drand48()) / NY as f32); // reverse it.
            let r = cam.get_ray(u, v);
            col = col + color(&r, &world, 0);
        }
        col = col / NS as f32;
        col = Vec3::new(col.x.sqrt(), col.y.sqrt(), col.z.sqrt()); // gamma 2
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

    write_png().unwrap()
}
