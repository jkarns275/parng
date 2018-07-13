#![feature(target_feature)]
#![feature(cfg_target_feature)]
#![feature(stdsimd)]

#[macro_use]
extern crate stdsimd;
extern crate rand;

pub mod portable;

#[cfg(test)]
mod tests {
    use portable::SFMTRng;
    use rand::{Rand, Rng, RngCore};
    use rand::ChaChaRng;

    #[test]
    fn profile_reference() {
        let mut rng = ChaChaRng::new_unseeded();
        const RANGE_LIMIT: u32 = 1024;

        use std::collections::HashMap;
        let mut counts: HashMap<u32, f64> = HashMap::new();

        for i in 0..1024 * 1024 {
            let r = rng.next_u32() % RANGE_LIMIT;
            *counts.entry(r).or_insert(0.0) += 1.0;
        }

        let mut sum = 0f64;
        for (k, v) in counts.iter() {
            sum += (*k as f64) * *v;
        }

        let average = sum / (1024 * 1024) as f64;

        let mut sum_squared_diff = 0.0;
        for (k, v) in counts.iter() {
            sum_squared_diff += *v * (*k as f64 - average).powi(2);
        }

        let variance = sum_squared_diff / (1024 * 1024 - 1) as f64;
        let std_dev = variance.sqrt();

        println!("reference variance: {}\nreference standard deviation: {}", variance, std_dev);
    }

    #[test]
    fn profile_sfmt() {
        let mut rng = SFMTRng::new(1129);
        const RANGE_LIMIT: u32 = 1024;

        use std::collections::HashMap;
        let mut counts: HashMap<u32, f64> = HashMap::new();

        for i in 0..1024 * 1024 {
            let r = rng.next_u32() % RANGE_LIMIT;
            *counts.entry(r).or_insert(0.0) += 1.0;
        }

        let mut sum = 0f64;
        for (k, v) in counts.iter() {
            sum += (*k as f64) * *v;
        }

        let average = sum / (1024 * 1024) as f64;

        let mut sum_squared_diff = 0.0;
        for (k, v) in counts.iter() {
            sum_squared_diff += *v * (*k as f64 - average).powi(2);
        }

        let variance = sum_squared_diff / (1024 * 1024 - 1) as f64;
        let std_dev = variance.sqrt();

        println!("sfmt variance: {}\nssfmt tandard deviation: {}", variance, std_dev);
    }
}
