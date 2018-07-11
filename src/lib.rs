#![feature(stdsimd)]

extern crate rand;

use std::simd;

pub mod portable;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
