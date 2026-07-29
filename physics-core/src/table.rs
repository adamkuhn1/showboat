//! Table geometry: origin at centre, long axis x, short axis y. Four straight
//! cushions and six pockets (four corner, two side).

use crate::constants::*;
use crate::vec::Vec2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CushionSide {
    Left,
    Right,
    Bottom,
    Top,
}

#[derive(Clone, Copy, Debug)]
pub struct Cushion {
    pub side: CushionSide,
    pub axis_x: bool, // true: fixed x coordinate; false: fixed y
    pub pos: f64,
    pub normal: Vec2, // inward normal
}

#[derive(Clone, Copy, Debug)]
pub struct Pocket {
    pub id: u8,
    pub center: Vec2,
    pub radius: f64,
}

#[derive(Clone, Debug)]
pub struct Table {
    pub length: f64,
    pub width: f64,
    pub cushions: [Cushion; 4],
    pub pockets: [Pocket; 6],
}

impl Table {
    pub fn bar_box() -> Self {
        let hx = TABLE_LENGTH / 2.0;
        let hy = TABLE_WIDTH / 2.0;
        Table {
            length: TABLE_LENGTH,
            width: TABLE_WIDTH,
            cushions: [
                Cushion { side: CushionSide::Left, axis_x: true, pos: -hx, normal: Vec2::new(1.0, 0.0) },
                Cushion { side: CushionSide::Right, axis_x: true, pos: hx, normal: Vec2::new(-1.0, 0.0) },
                Cushion { side: CushionSide::Bottom, axis_x: false, pos: -hy, normal: Vec2::new(0.0, 1.0) },
                Cushion { side: CushionSide::Top, axis_x: false, pos: hy, normal: Vec2::new(0.0, -1.0) },
            ],
            pockets: [
                Pocket { id: 0, center: Vec2::new(-hx, -hy), radius: CORNER_POCKET_RADIUS },
                Pocket { id: 1, center: Vec2::new(-hx, hy), radius: CORNER_POCKET_RADIUS },
                Pocket { id: 2, center: Vec2::new(hx, -hy), radius: CORNER_POCKET_RADIUS },
                Pocket { id: 3, center: Vec2::new(hx, hy), radius: CORNER_POCKET_RADIUS },
                Pocket { id: 4, center: Vec2::new(0.0, -hy), radius: SIDE_POCKET_RADIUS },
                Pocket { id: 5, center: Vec2::new(0.0, hy), radius: SIDE_POCKET_RADIUS },
            ],
        }
    }
}

pub fn cushion_side_str(s: CushionSide) -> &'static str {
    match s {
        CushionSide::Left => "left",
        CushionSide::Right => "right",
        CushionSide::Bottom => "bottom",
        CushionSide::Top => "top",
    }
}
