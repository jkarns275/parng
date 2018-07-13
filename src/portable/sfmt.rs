use stdsimd::simd::{u32x4, u8x16};
use std::mem::transmute;

use rand::{self, RngCore, Rng, SeedableRng, Rand};


/// The following implementation of the SIMD-oriented Fast Mersenne Twister algorithm is based on
/// the C++ reference implementation located (here)[http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/SFMT/].
///

/// The Mersenne Exponent. The period of the sequence will be a mutliple of 2^SFMT_MEXP - 1.
const SFMT_MEXP: usize = 19937;

/// The SFMT generator uses an internal state array of 128 bit integers - this is the size of the
/// array.
const SFMT_N128: usize = (SFMT_MEXP / 128) + 1;

/// The size of the SFMT generator internal state in terms of 8 bit integers.
const SFMT_N8: usize = SFMT_N128 * 16;

/// The size of the SFMT generator internal state in terms of 32 bit integers.
const SFMT_N32: usize = SFMT_N128 * 4;

/// The size of the SFMT generator internal state in terms of 64 bit integers.
const SFMT_N64: usize = SFMT_N128 * 2;

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
fn adjust(x: u32) -> usize { x as usize}


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
        let x = W128 { xx: a.xx << SFMT_SL2 };
        let y = W128 { xx: c.xx << SFMT_SR2 };
        if cfg!(target_endian = "little") {
            self.ex[0] = a.ex[0] ^ x.ex[0] ^ ((b.ex[0] >> SFMT_SR1) & SFMT_MSK1)
                ^ y.ex[0] ^ (d.ex[0] << SFMT_SL1);
            self.ex[1] = a.ex[1] ^ x.ex[1] ^ ((b.ex[1] >> SFMT_SR1) & SFMT_MSK2)
                ^ y.ex[1] ^ (d.ex[1] << SFMT_SL1);
            self.ex[2] = a.ex[2] ^ x.ex[2] ^ ((b.ex[2] >> SFMT_SR1) & SFMT_MSK3)
                ^ y.ex[2] ^ (d.ex[2] << SFMT_SL1);
            self.ex[3] = a.ex[3] ^ x.ex[3] ^ ((b.ex[3] >> SFMT_SR1) & SFMT_MSK4)
                ^ y.ex[3] ^ (d.ex[3] << SFMT_SL1); 
        } else {
            self.ex[0] = a.ex[0] ^ x.ex[0] ^ ((b.ex[0] >> SFMT_SR1) & SFMT_MSK2)
                ^ y.ex[0] ^ (d.ex[0] << SFMT_SL1);
            self.ex[1] = a.ex[1] ^ x.ex[1] ^ ((b.ex[1] >> SFMT_SR1) & SFMT_MSK1)
                ^ y.ex[1] ^ (d.ex[1] << SFMT_SL1);
            self.ex[2] = a.ex[2] ^ x.ex[2] ^ ((b.ex[2] >> SFMT_SR1) & SFMT_MSK4)
                ^ y.ex[2] ^ (d.ex[2] << SFMT_SL1);
            self.ex[3] = a.ex[3] ^ x.ex[3] ^ ((b.ex[3] >> SFMT_SR1) & SFMT_MSK3)
                ^ y.ex[3] ^ (d.ex[3] << SFMT_SL1);        
        }
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    pub fn do_recursion(&mut self, a: Self, b: Self, c: Self, d: Self) {
        //if is_x86_feature_detected!("sse2") {
        //    unsafe { self.sse_recursion(a, b, c, d) };
        //} else {
            unsafe { self.non_sse_recursion(a, b, c, d) };
        //}
    }
}

union SFMTState {
    w: [W128; SFMT_N128],
    l: [u64; SFMT_N64],
    u: [u32; SFMT_N32],
    b: [u8; SFMT_N8]
}

pub struct SFMTRng {
    /// The current index into the SFMTState - the index where the next random data will be read
    /// from.
    /// # Important
    /// `index` must be 4-byte aligned, since much of the code depends on that fact.
    index: usize,
    state: SFMTState,
}

impl SFMTRng {
    pub fn new_unseeded() -> Self {
        SFMTRng {
            index: usize::max_value(),
            state: SFMTState { w: [W128 { xx: 0u128 }; SFMT_N128] }
        }
    }

    pub fn new(seed: u32) -> Self {
        // Unsafe is used to turn the seed into an array of 4 bytes, which is then used as the seed.
        Self::from_seed(unsafe { transmute(seed.to_le()) })
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
                work = 1;
                for j in 0..32 {
                    if work & PARITY[i] != 0 {
                        state32[adjust(i as u32)] ^= work;
                        return
                    }
                    work <<= 1;
                }
            }
        }
    }

    /// Fills the entire internal state with random integers.
    fn fill_state_with_rand(&mut self) {
        println!("<<<<<<<<<<<<<<<<<<<<<<<<<<<Filling");
        unsafe {
            let mut i = 0;
            let mut r1 = self.state.w[SFMT_N128 - 2];
            let mut r2 = self.state.w[SFMT_N128 - 1];

            let state = &mut self.state.w[..];
            while i < (SFMT_N128 - SFMT_POS1 as usize) {
                let (a, b) = (*state.get_unchecked(i), *state.get_unchecked(i + SFMT_POS1 as usize));
                state.get_unchecked_mut(i)
                    .do_recursion(a, b, r1, r2);
                r1 = r2;
                r2 = *state.get_unchecked(i);
                i += 1;
            }
            while i < SFMT_N128 {
                let (a, b) = (*state.get_unchecked(i), *state.get_unchecked(i + SFMT_POS1 as usize - SFMT_N128));
                state.get_unchecked_mut(i).do_recursion(a, b, r1, r2);

                r1 = r2;
                r2 = *state.get_unchecked(i);
                i += 1;
            }
        }
    }
}

impl Rand for SFMTRng {
    fn rand<R: Rng>(rng: &mut R) -> Self {
        Self::new(rng.next_u32())
    }
}

impl RngCore for SFMTRng {
    fn next_u32(&mut self) -> u32 {
        if self.index >= SFMT_N32 {
            unsafe { self.fill_state_with_rand(); }
            self.index = 0;
        }
        let r = unsafe { self.state.u[self.index] };
        self.index += 1;
        r
    }

    fn next_u64(&mut self) -> u64 {
        // Ranges in match statements with the triple dot operator '...' are inclusive
        const SFMT_N32_MINUS_ONE: usize = SFMT_N32 - 1;
        match self.index {
            0 ... SFMT_N32_MINUS_ONE => {
                let r1 = unsafe { self.state.u[self.index] } as u64;
                self.index += 1;

                unsafe { self.fill_state_with_rand(); }
                self.index = 1;

                let r2 = unsafe { self.state.u[0] } as u64;
                self.index += 1;

                r1 as u64 | ((r2 as u64) << 32)
            },
            SFMT_N32 => {
                unsafe { self.fill_state_with_rand(); }
                let r = unsafe { self.state.l[self.index / 2] };
                self.index = 2;
                r
            },
            _ => {
                let r = unsafe { self.state.l[self.index / 2] };
                self.index += 2;
                r
            }
        }
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        let original_len = dest.len();
        let whole_u32s = dest.len() / 4;
        // The length in bytes of the number of whole u32's in dst. The up to 3 remaining bytes will
        // be filled by the fill_bytes outer function
        let truncated_len = whole_u32s * 4;
        let mut dest = dest;

        while dest.len() > 0 {
            {
                let mut data = unsafe { &self.state.b[self.index..] };
                if data.len() > dest.len() {
                    data = &data[0..dest.len()];
                    dest.copy_from_slice(data);
                    let temp = dest;
                    dest = &mut temp[data.len()..];
                    self.index += data.len() / 4
                        // Accounts for the fact that dest.len() might not be for-byte aligned to keep index 4 byte aligned
                        + (data.len() & 3 == 0) as usize;
                    continue;
                } else {
                    dest[0..data.len()].copy_from_slice(data);
                    let temp = dest;
                    dest = &mut temp[data.len()..];
                }
            }
            self.fill_state_with_rand();
            self.index = 0;
        }

        if self.index >= SFMT_N32 {
            self.fill_state_with_rand();
            self.index = 0;
        }
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand::Error> {
        self.fill_bytes(dest);
        return Ok(())
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
                    FACTOR.wrapping_mul(*state32.get_unchecked(adjust(i - 1))
                              ^ (*state32.get_unchecked(adjust(i - 1)) >> 30)).wrapping_add(i);
                i += 1;
            }
        }
        unsafe { result.fill_state_with_rand() };
        result.index = SFMT_N32;
        result.period_certification();
        result
    }
}
