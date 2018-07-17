use rand::{RngCore, Rand, SeedableRng};
use stdsimd::simd::{u64x2, u8x16};

#[cfg(target_arch = "x86")]
use stdsimd::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use stdsimd::arch::x86_64::*;

#[derive(Copy, Clone)]
#[repr(align(16))]
union XorShift128 {
    xx: u128,
    mx: u64x2,
    ix: __m128i,
    rx: [u64; 2],
    ex: [u32; 4],
    tx: (u64, u64)
}

struct XorShiftState([XorShift128; 3]);

impl XorShiftState {

    #[inline(always)]
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    pub unsafe fn gen_sse(&mut self) {
        let mut x = _mm_load_si128(&mut self.0[0].ix as *const __m128i);
        let y = _mm_load_si128(&mut self.0[1].ix as *const __m128i);
        _mm_store_si128(&mut self.0[0].ix as *mut __m128i, y);
        x = _mm_xor_si128(x, _mm_slli_epi64(x, 23));
        self.0[1].ix = _mm_xor_si128(_mm_xor_si128(y, _mm_srli_epi64(y, 26)),
                                     _mm_xor_si128(x, _mm_srli_epi64(x, 17)));
        self.0[2].ix = _mm_add_epi64(self.0[1].ix, y);
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    pub unsafe fn gen_non_sse(&mut self) {
        let (mut x, mut z) = self.0[0].tx;
        let (y, w) = self.0[1].tx;
        self.0[0].tx = (y, w);
        x ^= x << 23;
        z ^= z << 23;
        self.0[1].tx = (x ^ y ^ (x >> 17) ^ (y >> 26), z ^ w ^ (z >> 16) ^ (w >> 26));
        self.0[2].tx = (self.0[1].tx.0 + y, self.0[1].tx.1 + w);
    }
}