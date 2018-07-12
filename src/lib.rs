#![feature(target_feature)]
#![feature(cfg_target_feature)]
#![feature(stdsimd)]

#[macro_use]
extern crate stdsimd;
extern crate rand;

pub mod portable;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
