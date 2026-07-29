//! 2D vector math for the top-down pool plane. Spin about the vertical axis is
//! tracked separately on the ball; these are the in-plane operations.

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vec2 {
    pub x: f64,
    pub y: f64,
}

impl Vec2 {
    pub const ZERO: Vec2 = Vec2 { x: 0.0, y: 0.0 };

    #[inline]
    pub fn new(x: f64, y: f64) -> Self {
        Vec2 { x, y }
    }

    #[inline]
    pub fn add(self, o: Vec2) -> Vec2 {
        Vec2::new(self.x + o.x, self.y + o.y)
    }

    #[inline]
    pub fn sub(self, o: Vec2) -> Vec2 {
        Vec2::new(self.x - o.x, self.y - o.y)
    }

    #[inline]
    pub fn scale(self, s: f64) -> Vec2 {
        Vec2::new(self.x * s, self.y * s)
    }

    #[inline]
    pub fn dot(self, o: Vec2) -> f64 {
        self.x * o.x + self.y * o.y
    }

    #[inline]
    pub fn mag(self) -> f64 {
        self.x.hypot(self.y)
    }

    #[inline]
    pub fn normalize(self) -> Vec2 {
        let m = self.mag();
        if m < 1e-12 {
            Vec2::ZERO
        } else {
            Vec2::new(self.x / m, self.y / m)
        }
    }

    /// Left-hand perpendicular (rotate +90 degrees).
    #[inline]
    pub fn perp(self) -> Vec2 {
        Vec2::new(-self.y, self.x)
    }

    #[inline]
    pub fn from_angle(theta: f64, m: f64) -> Vec2 {
        Vec2::new(theta.cos() * m, theta.sin() * m)
    }
}
