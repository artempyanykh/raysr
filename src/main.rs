use std::{
    fmt::{Debug, Display},
    fs::File,
    io::{BufWriter, Write},
    ops::{Add, Div, Mul, Sub},
};

fn main() -> std::io::Result<()> {
    let mut buf = BufWriter::new(File::create("sample.ppm")?);
    let img = basic_scene();
    write!(buf, "{}", PPM(&img))?;
    buf.flush()?;
    println!("Successfully generated an image");
    Ok(())
}

fn sample_image() -> Image {
    let width = 256;
    let height = 256;
    Image::new_with_init(height, width, |row, col| {
        let r = row as f64 / ((height - 1) as f64);
        let g = col as f64 / ((width - 1) as f64);
        let b = 0.1;

        Color::new(r, g, b).to_pixel()
    })
}

fn ray_color(ray: &Ray) -> Color {
    let unit_dir = ray.dir.unit();
    let blend_factor = 0.5 * (unit_dir.y + 1.);
    (1. - blend_factor) * Color::new(1., 1., 1.) + blend_factor * Color::new(0.5, 0.7, 1.)
}

fn basic_scene() -> Image {
    let aspect_ratio = 16. / 9.;
    let img_width: usize = 400;
    let img_height = (img_width as f64 / aspect_ratio) as usize;

    // Camera
    let focal_len = 1.;
    let viewport_height = 2.;
    let viewport_width = viewport_height * (img_width as f64) / (img_height as f64);
    let camera_center = Point3::new(0., 0., 0.);

    // Calculate viewport vector (left-right, up-down)
    let viewport_lr = Vec3::new(viewport_width, 0., 0.);
    let viewport_ud = Vec3::new(0., -viewport_height, 0.);

    // Calculate delta vectors from pixel to pixel
    let pixel_delta_lr = &viewport_lr / img_width;
    let pixel_delta_ud = &viewport_ud / img_height;

    // Calculate the location of the upper left pixel
    let viewport_upper_left =
        &camera_center - Vec3::new(0., 0., focal_len) - &viewport_lr / 2 - &viewport_ud / 2;
    let pixel00_loc = &viewport_upper_left + 0.5 * (&pixel_delta_lr + &pixel_delta_ud);

    // Render
    Image::new_with_init(img_height, img_width, |row, col| {
        let pixel_center = &pixel00_loc + row * &pixel_delta_ud + col * &pixel_delta_lr;
        let ray_dir = &pixel_center - &camera_center;
        let ray = Ray::new(camera_center.clone(), ray_dir);

        let color = ray_color(&ray);
        color.to_pixel()
    })
}

struct PPM<'a, T>(&'a T);

#[derive(Default)]
struct Pixel {
    r: u8,
    g: u8,
    b: u8,
}

impl Pixel {
    pub fn new(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b }
    }
}

impl Display for PPM<'_, Pixel> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:>3} {:>3} {:>3}", self.0.r, self.0.g, self.0.b)
    }
}

impl Debug for PPM<'_, Pixel> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self)
    }
}

struct Image {
    pixels: Vec<Vec<Pixel>>,
}

impl Image {
    pub fn new(height: usize, width: usize) -> Image {
        assert!(height > 0 && width > 0);

        let mut pixels = Vec::with_capacity(height);
        for _ in 0..height {
            let mut row = Vec::with_capacity(width);
            for _ in 0..width {
                row.push(Pixel::default());
            }
            pixels.push(row);
        }

        Self { pixels }
    }

    pub fn new_with_init(
        height: usize,
        width: usize,
        init: impl Fn(usize, usize) -> Pixel,
    ) -> Image {
        let mut image = Self::new(height, width);
        for row in 0..height {
            for col in 0..width {
                image.pixels[row][col] = init(row, col);
            }
        }

        image
    }

    pub fn num_rows(&self) -> usize {
        self.pixels.len()
    }

    pub fn num_cols(&self) -> usize {
        self.pixels[0].len()
    }
}

impl Display for PPM<'_, Image> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "P3\n{} {}\n255", self.0.num_cols(), self.0.num_rows())?;

        for row in self.0.pixels.iter() {
            for pixel in row.iter() {
                writeln!(f, "{}", PPM(pixel))?;
            }
        }

        Ok(())
    }
}

impl Debug for PPM<'_, Image> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self)
    }
}

#[derive(Clone)]
struct Vec3 {
    x: f64,
    y: f64,
    z: f64,
}

impl Vec3 {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    pub fn len_squared(&self) -> f64 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    pub fn len(&self) -> f64 {
        self.len_squared().sqrt()
    }

    pub fn unit(&self) -> Self {
        self / self.len()
    }
}

impl Debug for Vec3 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {}, {})", self.x, self.y, self.z)
    }
}

// https://github.com/rust-lang/rust/issues/44762
// https://lang-team.rust-lang.org/design_notes/autoref_ops.html
impl Add<&Vec3> for &Vec3 {
    type Output = Vec3;

    fn add(self, rhs: &Vec3) -> Self::Output {
        Vec3::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl Add<Vec3> for &Vec3 {
    type Output = Vec3;

    fn add(self, rhs: Vec3) -> Self::Output {
        self + &rhs
    }
}

impl Add<&Vec3> for Vec3 {
    type Output = Self;

    fn add(self, rhs: &Vec3) -> Self::Output {
        &self + rhs
    }
}

impl Add for Vec3 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        self + &rhs
    }
}

impl Sub<&Vec3> for &Vec3 {
    type Output = Vec3;

    fn sub(self, rhs: &Vec3) -> Self::Output {
        Vec3::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

impl Sub<Vec3> for &Vec3 {
    type Output = Vec3;

    fn sub(self, rhs: Vec3) -> Self::Output {
        self - &rhs
    }
}

impl Sub<&Vec3> for Vec3 {
    type Output = Self;

    fn sub(self, rhs: &Vec3) -> Self::Output {
        &self - rhs
    }
}

impl Sub for Vec3 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self - &rhs
    }
}

impl Mul<&Vec3> for f64 {
    type Output = Vec3;

    fn mul(self, rhs: &Vec3) -> Self::Output {
        Self::Output {
            x: self * rhs.x,
            y: self * rhs.y,
            z: self * rhs.z,
        }
    }
}

impl Mul<Vec3> for f64 {
    type Output = Vec3;

    fn mul(self, rhs: Vec3) -> Self::Output {
        self * &rhs
    }
}

impl Mul<&Vec3> for usize {
    type Output = Vec3;

    fn mul(self, rhs: &Vec3) -> Self::Output {
        self as f64 * rhs
    }
}

impl Div<f64> for &Vec3 {
    type Output = Vec3;

    fn div(self, rhs: f64) -> Self::Output {
        Self::Output::new(self.x / rhs, self.y / rhs, self.z / rhs)
    }
}

impl Div<f64> for Vec3 {
    type Output = Self;

    fn div(self, rhs: f64) -> Self::Output {
        &self / rhs
    }
}

impl Div<usize> for &Vec3 {
    type Output = Vec3;

    fn div(self, rhs: usize) -> Self::Output {
        self / (rhs as f64)
    }
}

impl Div<usize> for Vec3 {
    type Output = Self;

    fn div(self, rhs: usize) -> Self::Output {
        &self / (rhs as f64)
    }
}

type Color = Vec3;

impl Color {
    pub fn to_pixel(&self) -> Pixel {
        let factor = 255.999;
        let scaled = factor * self;
        Pixel::new(scaled.x as u8, scaled.y as u8, scaled.z as u8)
    }
}

type Point3 = Vec3;

struct Ray {
    pub origin: Point3,
    pub dir: Vec3,
}

impl Ray {
    pub fn new(origin: Point3, dir: Vec3) -> Self {
        Self { origin, dir }
    }

    pub fn at(&self, scale: f64) -> Self {
        Self::new(self.origin.clone(), scale * &self.dir)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test1() {
        let img =
            Image::new_with_init(2, 3, |row, col| Pixel::new((row + col) as u8, col as u8, 0));
        k9::snapshot!(
            PPM(&img),
            "
P3
3 2
255
  0   0   0
  1   1   0
  2   2   0
  1   0   0
  2   1   0
  3   2   0

"
        );
    }

    #[test]
    fn test_vec_ops() {
        let v1 = Vec3::new(1., 2., 3.);
        let v2 = Vec3::new(4., 5., 6.);

        k9::snapshot!(v1 + v2, "(5, 7, 9)");

        let v1 = Vec3::new(1., 2., 3.);
        let v2 = &Vec3::new(4., 5., 6.);
        k9::snapshot!(v1 + v2, "(5, 7, 9)");

        let v1 = &Vec3::new(1., 2., 3.);
        let v2 = &Vec3::new(4., 5., 6.);
        k9::snapshot!(v1 + v2, "(5, 7, 9)");
    }

    #[test]
    fn test_color() {
        let real_color = Color::new(0.1, 0.5, 0.9);
        k9::snapshot!(PPM(&real_color.to_pixel()), " 25 127 230");
    }
}
