#[macro_use]
use std::simd::cfg_feature_enabled;
use std::simd::{u32x4, u8x16};
use std::mem::transmute;

use rand::{self, RngCore, Rng, SeedableRng, Rand};


/// The following implementation of the SIMD-oriented Fast Mersenne Twister algorithm is based on
/// the C++ reference implementation located (here)[http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/SFMT/].
///

/// The Mersenne Exponent. The period of the sequence will be a mutliple of 2^SFMT_MEXP - 1.
const SFMT_MEXP: usize = 19937;

/// The SFMT generator uses an internal state array of 128 bit integers - this is the size of the
/// array.
const SFMT_N: usize = (SFMT_MEXP / 128) + 1;

/// The size of the SFMT generator internal state in terms of 32 bit integers.
const SFMT_N32: usize = SFMT_N * 4;

/// The size of the SFMT generator internal state in terms of 64 bit integers.
const SFMT_N64: usize = SFMT_N * 2;

/*
 * The following constants are parameters to SFMT. These constants are specific to the selected
 * Mersenne Exponent (in this case, SFMT_MEXP is 19937), taken from the C++ implemetation by
 * M. Saito et al. I'm not sure what would happen if the
 * exponent were changed, but these numbers were left.
 */

const SFMT_POS1: u32 = 122;
const SFMT_SL1: u32	= 18;
const SFMT_SL2: u32 = 1;
const SFMT_SR1: u32 = 11;
const SFMT_SR2: u32 = 1;
const SFMT_MSK1: u32 = 0xdfffffef;
const SFMT_MSK2: u32 = 0xddfecb7f;
const SFMT_MSK3: u32 = 0xbffaffff;
const SFMT_MSK4: u32 = 0xbffffff6;
const SFMT_PARITY1: u32 = 0x00000001;
const SFMT_PARITY2: u32 = 0x00000000;
const SFMT_PARITY3: u32 = 0x00000000;
const SFMT_PARITY4: u32 = 0x13c9e684;
const SFMT_ALTI_SL1: u32x4 =        u32x4::new(SFMT_SL1, SFMT_SL1, SFMT_SL1, SFMT_SL1);
const SFMT_ALTI_SR1: u32x4 =        u32x4::new(SFMT_SR1, SFMT_SR1, SFMT_SR1, SFMT_SR1);
const SFMT_ALTI_MSK: u32x4 =        u32x4::new(SFMT_MSK1, SFMT_MSK2, SFMT_MSK3, SFMT_MSK4);
const SFMT_ALTI_MSK64: u32x4 =      u32x4::new(SFMT_MSK2, SFMT_MSK1, SFMT_MSK4, SFMT_MSK3);
const SFMT_ALTI_SL2_PERM: u8x16 =   u8x16::new(1,2,3,23,5,6,7,0,9,10,11,4,13,14,15,8);
const SFMT_ALTI_SL2_PERM64: u8x16 = u8x16::new(1,2,3,4,5,6,7,31,9,10,11,12,13,14,15,0);
const SFMT_ALTI_SR2_PERM: u8x16 =   u8x16::new(7,0,1,2,11,4,5,6,15,8,9,10,17,12,13,14);
const SFMT_ALTI_SR2_PERM64: u8x16 = u8x16::new(15,0,1,2,3,4,5,6,17,8,9,10,11,12,13,14);

/// Helper function to ensure code is endian-neutral
#[inline(always)]
#[cfg(target_endian = "big")]
fn adjust(x: u32) -> usize { x as usize ^ 1 }

/// Helper function to ensure code is endian-neutral
#[inline(always)]
#[cfg(target_endian = "little")]
fn adjust(x: u32) -> usize { x as usize }


#[derive(Copy, Clone)]
union W128 {
    xx: u128,
    mx: u32x4,
    rx: [u64; 2],
    ex: [u32; 4]
}

impl W128 {
    #[inline(always)]
    unsafe fn sse_recursion(&mut self, a: Self, b: Self, c: Self, d: Self) {
        const SSE2_PARAM_MASK: W128 = W128 { ex: [SFMT_MSK1, SFMT_MSK2, SFMT_MSK3, SFMT_MSK4] };

        let (mut v, mut x, mut y, mut z) =
            (W128 { xx: 0 }, W128 { xx: 0 }, W128 { xx: 0 }, W128 { xx: 0 });

        y.mx = b.mx >> SFMT_SR1;
        z.xx = c.xx >> SFMT_SR2;
        v.mx = d.mx >> SFMT_SL1;
        z.xx ^= a.xx;
        z.xx ^= v.xx;
        x.xx = a.xx << SFMT_SL2;
        y.xx &= SSE2_PARAM_MASK.xx;
        z.xx ^= x.xx;
        z.xx ^= y.xx;

        self.xx = z.xx;
    }

    #[inline(always)]
    unsafe fn non_sse_recursion(&mut self, a: Self, b: Self, c: Self, d: Self) {
        let x = W128 { xx: a << SFMT_SL2 };
        let y = W128 { xx: c << SFMT_SR2 };
        if cfg!(target_endian = "little") {
            self.ex[0] = a.ex[0] ^ x.ex[0] ^ ((b.ex[0] >> SFMT_SR1) & SFMT_MSK1)
                ^ y.ex[0] ^ (d.u[0] << SFMT_SL1);
            self.ex[1] = a.u[1] ^ x.ex[1] ^ ((b.ex[1] >> SFMT_SR1) & SFMT_MSK2)
                ^ y.ex[1] ^ (d.u[1] << SFMT_SL1);
            self.ex[2] = a.ex[2] ^ x.ex[2] ^ ((b.ex[2] >> SFMT_SR1) & SFMT_MSK3)
                ^ y.ex[2] ^ (d.ex[2] << SFMT_SL1);
            self.ex[3] = a.ex[3] ^ x.ex[3] ^ ((b.ex[3] >> SFMT_SR1) & SFMT_MSK4)
                ^ y.ex[3] ^ (d.ex[3] << SFMT_SL1); 
        } else {
            self.ex[0] = a.ex[0] ^ x.ex[0] ^ ((b.ex[0] >> SFMT_SR1) & SFMT_MSK2)
                ^ y.ex[0] ^ (d.u[0] << SFMT_SL1);
            self.ex[1] = a.u[1] ^ x.ex[1] ^ ((b.ex[1] >> SFMT_SR1) & SFMT_MSK1)
                ^ y.ex[1] ^ (d.u[1] << SFMT_SL1);
            self.ex[2] = a.ex[2] ^ x.ex[2] ^ ((b.ex[2] >> SFMT_SR1) & SFMT_MSK4)
                ^ y.ex[2] ^ (d.ex[2] << SFMT_SL1);
            self.ex[3] = a.ex[3] ^ x.ex[3] ^ ((b.ex[3] >> SFMT_SR1) & SFMT_MSK3)
                ^ y.ex[3] ^ (d.ex[3] << SFMT_SL1);        
        }
    }

    pub fn do_recursion(&mut self, a: Self, b: Self, c: Self, d: Self) {
        if cfg_feature_enabled!("sse2") {
            self.sse_recursion(a, b, c, d);    
        } else {
            self.non_sse_recursion(a, b, c, d);
        }
    }
}

union SFMTState {
    w: [W128; SFMT_N],
    l: [u64; SFMT_N64],
    u: [u32; SFMT_N32]
}

pub struct SFMTRng {
    index: usize,
    state: SFMTState,
}

impl SFMTRng {
    pub fn new_unseeded() -> Self {
        SFMTRng {
            index: 0,
            state: SFMTState { w: [W128 { xx: 0u128 }; SFMT_N] }
        }
    }

    pub fn new(seed: u32) -> Self {
        // Unsafe is used to turn the seed into an array of 4 bytes, which is then used as the seed.
        Self::from_seed(unsafe { transmute(seed) })
    }

    /// Guarantees that the period of the series is 2^SFMT_MEXP
    fn period_certification(&mut self) {
        const PARITY: [u32; 4] = [SFMT_PARITY1, SFMT_PARITY2, SFMT_PARITY3, SFMT_PARITY4];

        unsafe {
            let mut inner_product = 0u32;
            let mut work = 0u32;

            let state32 = unsafe { &mut self.state.u[..] };
            
            if cfg!(target_endian = "little") {
                inner_product ^= *state32.get_unchecked(0) & PARITY[0];
                inner_product ^= *state32.get_unchecked(1) & PARITY[1];
                inner_product ^= *state32.get_unchecked(2) & PARITY[2];
                inner_product ^= *state32.get_unchecked(3) & PARITY[3];
            } else {
                inner_product ^= *state32.get_unchecked(1) & PARITY[0];
                inner_product ^= *state32.get_unchecked(0) & PARITY[1];
                inner_product ^= *state32.get_unchecked(3) & PARITY[2];
                inner_product ^= *state32.get_unchecked(2) & PARITY[3];
            }

            let mut i = 16;
            inner_product ^= inner_product >> 16;
            inner_product ^= inner_product >> 8;
            inner_product ^= inner_product >> 4;
            inner_product ^= inner_product >> 1;

            inner_product &= 1;

            /* Check OK  */
            if inner_product == 1 { return }

            for i in 0..4 {
                work = 0;
                for j in 0..32 {
                    if work & PARITY[i] != 0 {
                        state32[adjust(i)] ^= work;
                        return
                    }
                    work <<= 1;
                }
            }
        }
    }

    /// Fills the entire internal state with random integers.
    fn fill_state_with_rand(&mut self) {
        let mut i = 0;
        let mut r1 = W128 { xx: 0 };
        let mut r2 = W128 { xx: 0 };

        let state = &mut self.state.w[..];
        while i < (SFMT_N - SFMT_POS1) {
            state[i].do_recursion(state[i], state[i + SFMT_POS1], r1, r2);
            r1 = r2;
            r2 = state[i];
            i += 1;
        }
        while i < SFMT_N {
            state[i].do_recursion(state[i], state[i + SFMT_POS1 - SFMT_N], r1, r2);
            r1 = r2;
            r2 = state[i];
            i += 1;
        }
    }
    
}

impl Rand for SFMTRng {
    fn rand<R: Rng>(rng: &mut R) -> Self {
        unimplemented!()
    }
}

impl RngCore for SFMTRng {
    fn next_u32(&mut self) -> u32 {
        unimplemented!()
    }

    fn next_u64(&mut self) -> u64 {
        unimplemented!()
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        unimplemented!()
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand::Error> {
        unimplemented!()
    }
}

impl SeedableRng for SFMTRng {
    type Seed = [u8; 4];

    fn from_seed(seed: <Self as SeedableRng>::Seed) -> Self {
        const FACTOR: u32 = 1812433253;

        let seed: u32 = u32::from_le(unsafe { transmute(seed) });
        let mut result = Self::new_unseeded();
        
        unsafe {
            let state32 = &mut result.state.u[..];
            state32[adjust(0)] = seed;

            let mut i = 1u32;
            while i < SFMT_N32 as u32 {
                let new_value =
                    FACTOR * (*state32.get_unchecked(adjust(i - 1))
                              ^ (*state32.get_unchecked(adjust(i - 1)) >> 30)) + i;
                i += 1;
            }
        }
        result.index = SFMT_N32;
        result.period_certification();
        result
    }
}
