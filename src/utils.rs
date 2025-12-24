pub fn is_prime_i64(n: i64) -> bool {
    // i64::MIN cannot be negated; and it's even anyway, so not prime.
    if n == i64::MIN {
        return false;
    }

    let n = n.abs();

    if n < 2 {
        return false;
    }
    if n == 2 || n == 3 {
        return true;
    }
    if n % 2 == 0 || n % 3 == 0 {
        return false;
    }

    // 6k ± 1 wheel: 5, 7, 11, 13, 17, 19, ...
    let mut i: i64 = 5;
    let mut step: i64 = 2;

    // Avoid overflow via division
    while i <= n / i {
        if n % i == 0 {
            return false;
        }
        i += step;
        step = 6 - step;
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_small_primes() {
        assert!(is_prime_i64(2));
        assert!(is_prime_i64(3));
        assert!(is_prime_i64(5));
        assert!(is_prime_i64(7));
        assert!(is_prime_i64(11));
        assert!(is_prime_i64(13));
        assert!(is_prime_i64(17));
        assert!(is_prime_i64(19));
        assert!(is_prime_i64(23));
    }

    #[test]
    fn test_small_composites() {
        assert!(!is_prime_i64(4));
        assert!(!is_prime_i64(6));
        assert!(!is_prime_i64(8));
        assert!(!is_prime_i64(9));
        assert!(!is_prime_i64(10));
        assert!(!is_prime_i64(12));
        assert!(!is_prime_i64(15));
        assert!(!is_prime_i64(21));
    }

    #[test]
    fn test_edge_cases() {
        assert!(!is_prime_i64(0));
        assert!(!is_prime_i64(1));
        assert!(!is_prime_i64(-1));
        assert!(!is_prime_i64(i64::MIN));
    }

    #[test]
    fn test_negative_primes() {
        // abs() of negative primes should still return true
        assert!(is_prime_i64(-2));
        assert!(is_prime_i64(-3));
        assert!(is_prime_i64(-5));
        assert!(is_prime_i64(-7));
    }

    #[test]
    fn test_larger_primes() {
        assert!(is_prime_i64(97));
        assert!(is_prime_i64(101));
        assert!(is_prime_i64(1009));
    }

    #[test]
    fn test_larger_composites() {
        assert!(!is_prime_i64(100));
        assert!(!is_prime_i64(1000));
        assert!(!is_prime_i64(49)); // 7*7
        assert!(!is_prime_i64(121)); // 11*11
    }
}
