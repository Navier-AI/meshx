/// Helper trait to interpret a slice of bytes representing a sized type.
///
/// Note: this trait is meant to be used transiently on the same platform, and so doesn't care
/// about endianness. In other words, the slice of bytes generated by this trait should not be
/// stored anywhere that outlives the lifetime of the program.
pub trait Bytes
where
    Self: Sized,
{
    /// Get a slice of bytes representing `Self`.
    #[inline]
    fn as_bytes(&self) -> &[u8] {
        let byte_ptr = self as *const Self as *const u8;
        unsafe { std::slice::from_raw_parts(byte_ptr, std::mem::size_of::<Self>()) }
    }

    /// Panics if the size of the given bytes slice is not equal to the size of `Self`.
    #[inline]
    fn interpret_bytes(bytes: &[u8]) -> &Self {
        assert_eq!(bytes.len(), std::mem::size_of::<Self>());
        let ptr = bytes.as_ptr() as *const Self;
        unsafe { &*ptr }
    }
}

impl<T: Sized> Bytes for T {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bytes_from_value_test() {
        let val = 10.2_f64;
        assert_eq!(&val, Bytes::interpret_bytes(val.as_bytes()));
    }
}
