// 2D vector helpers. The physics engine works in a top-down plane (x, y) in
// metres; the third rotational axis (spin about the vertical) is tracked
// separately on the ball. Kept as small pure functions so the hot paths stay
// allocation-light and easy to reason about in a review.

export interface Vec2 {
  x: number;
  y: number;
}

export const vec = (x: number, y: number): Vec2 => ({ x, y });

export const add = (a: Vec2, b: Vec2): Vec2 => ({ x: a.x + b.x, y: a.y + b.y });
export const sub = (a: Vec2, b: Vec2): Vec2 => ({ x: a.x - b.x, y: a.y - b.y });
export const scale = (a: Vec2, s: number): Vec2 => ({ x: a.x * s, y: a.y * s });
export const dot = (a: Vec2, b: Vec2): number => a.x * b.x + a.y * b.y;

// 2D cross product returns the scalar z-component.
export const cross = (a: Vec2, b: Vec2): number => a.x * b.y - a.y * b.x;

export const mag = (a: Vec2): number => Math.hypot(a.x, a.y);
export const mag2 = (a: Vec2): number => a.x * a.x + a.y * a.y;

export const normalize = (a: Vec2): Vec2 => {
  const m = mag(a);
  return m < 1e-12 ? { x: 0, y: 0 } : { x: a.x / m, y: a.y / m };
};

// Left-hand perpendicular (rotate +90 degrees).
export const perp = (a: Vec2): Vec2 => ({ x: -a.y, y: a.x });

export const fromAngle = (theta: number, m = 1): Vec2 => ({
  x: Math.cos(theta) * m,
  y: Math.sin(theta) * m,
});

export const angleOf = (a: Vec2): number => Math.atan2(a.y, a.x);
