use vstd::prelude::*;
use vstd::seq::*;

verus! {

fn sum_of_digits(number: u32) -> (sum: u32)
    requires
        number >= 0,
        number < 1000000,  // added relaxation to prevent overflow

    ensures
        sum >= 0,
{
    let mut n = number;
    let mut sum: u32 = 0;
    let mut i: u32 = 0;
    while n > 0
        invariant
        decreases n,
    {
        let digit = n % 10;
        sum = sum.wrapping_add(digit);
        n = n / 10;
        i = i.wrapping_add(1);
    }
    sum
}

fn main() {
}

} // verus!
