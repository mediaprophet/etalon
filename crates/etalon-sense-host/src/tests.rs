use super::*;
use std::cmp::min;
use std::io::ErrorKind;

struct ChunkedReader<'a> {
    bytes: &'a [u8],
    position: usize,
    max_chunk: usize,
    calls: usize,
    interrupt_call: Option<usize>,
}

impl<'a> ChunkedReader<'a> {
    fn new(bytes: &'a [u8], max_chunk: usize, interrupt_call: Option<usize>) -> Self {
        Self { bytes, position: 0, max_chunk, calls: 0, interrupt_call }
    }
}

impl Read for ChunkedReader<'_> {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        if self.interrupt_call == Some(self.calls) {
            self.calls += 1;
            return Err(io::Error::from(ErrorKind::Interrupted));
        }
        self.calls += 1;
        if self.position == self.bytes.len() {
            return Ok(0);
        }
        let count = min(min(out.len(), self.max_chunk), self.bytes.len() - self.position);
        out[..count].copy_from_slice(&self.bytes[self.position..self.position + count]);
        self.position += count;
        Ok(count)
    }
}

#[test]
fn bounded_read_handles_exact_fit_and_excess() {
    let mut exact = [0; 3];
    assert_eq!(read_bounded(&mut &b"abc"[..], &mut exact).unwrap(), 3);
    assert_eq!(&exact, b"abc");
    let mut short = [0; 2];
    assert!(matches!(read_bounded(&mut &b"abc"[..], &mut short), Err(InputError::Capacity)));
}

#[test]
fn empty_buffer_only_accepts_empty_input() {
    let mut empty = [];
    assert_eq!(read_bounded(&mut &b""[..], &mut empty).unwrap(), 0);
    assert!(matches!(read_bounded(&mut &b"x"[..], &mut empty), Err(InputError::Capacity)));
}

#[test]
fn bounded_read_retries_interrupted_and_accumulates_partial_reads() {
    let mut reader = ChunkedReader::new(b"abcdef", 2, Some(0));
    let mut buffer = [0; 6];
    assert_eq!(read_bounded(&mut reader, &mut buffer).unwrap(), buffer.len());
    assert_eq!(&buffer, b"abcdef");
}

#[test]
fn capacity_probe_retries_interrupted_then_rejects_extra_byte() {
    let mut reader = ChunkedReader::new(b"abc", 2, None);
    let mut buffer = [0; 2];
    assert!(matches!(read_bounded(&mut reader, &mut buffer), Err(InputError::Capacity)));
    assert_eq!(&buffer, b"ab");

    let mut interrupted_probe = ChunkedReader::new(b"abc", 2, Some(1));
    let mut exact = [0; 2];
    assert!(matches!(read_bounded(&mut interrupted_probe, &mut exact), Err(InputError::Capacity)));
    assert_eq!(&exact, b"ab");
}
