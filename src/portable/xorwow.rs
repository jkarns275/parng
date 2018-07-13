use rand::{RngCore, Rand, SeedableRng};
use stdsimd::simd::{u32x4, u8x16};

#[derive(Copy, Clone)]
union XorWow128 {
    xx: u128,
    mx: u32x4,
    rx: [u64; 2],
    ex: [u32; 4]
}

struct XorWowState {

}