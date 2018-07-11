#![feature(stdsimd)]
#![feature(target_feature)]
#![feature(cfg_target_feature)]
extern crate rand;

#[macro_use] extern crate stdsimd;

use std::simd;

pub mod portable;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
