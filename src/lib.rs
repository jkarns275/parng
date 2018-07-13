#![feature(target_feature)]
#![feature(cfg_target_feature)]
#![feature(stdsimd)]
#![feature(duration_as_u128)]

#[macro_use]
extern crate stdsimd;
extern crate rand;

pub mod portable;

#[cfg(test)]
mod tests {
    use portable::*;
    use rand::{Rand, Rng, RngCore};
    use rand::ChaChaRng;
    use std::time::Instant;

    #[test]
    fn profile_reference() {
        let start = Instant::now();
        let mut rng = ChaChaRng::new_unseeded();
        const RANGE_LIMIT: u32 = 1024;

        use std::collections::HashMap;
        let mut counts: HashMap<u32, f64> = HashMap::new();

        let mut numbers = Vec::<u32>::new();
        for i in 0..1024 * 1024 { numbers.push(rng.next_u32()) }
        let elapsed = start.elapsed();
        let duration = (elapsed.as_nanos() as f64) / 1e9;

        for r in numbers {
            *counts.entry(r % RANGE_LIMIT).or_insert(0.0) += 1.0;
        }
        let elapsed = 0;

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
        println!("reference variance: {}\nreference standard deviation: {}\nreference time: {}", variance, std_dev, duration);
   }

    #[test]
    fn profile_sfmt() {
        let start = Instant::now();
        let mut rng = SFMTRng::new(1129);
        const RANGE_LIMIT: u32 = 1024;

        use std::collections::HashMap;
        let mut counts: HashMap<u32, f64> = HashMap::new();

        let mut numbers = Vec::<u32>::new();
        for i in 0..1024 * 1024 { numbers.push(rng.next_u32()) }
        let elapsed = start.elapsed();
        let duration = (elapsed.as_nanos() as f64) / 1e9;

        for r in numbers {
            *counts.entry(r % RANGE_LIMIT).or_insert(0.0) += 1.0;
        }
        let elapsed = 0;

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
        println!("sfmt variance: {}\nsfmt standard deviation: {}\nsfmt time: {}", variance, std_dev, duration);
    }

    #[test]
    fn angery() {
        test1();
        test2();
    }
}
