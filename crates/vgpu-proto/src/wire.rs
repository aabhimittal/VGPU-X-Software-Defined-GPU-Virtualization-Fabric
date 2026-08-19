//! Byte-level primitives: cursored encode/decode and length-prefixed
//! framing.
//!
//! # Why hand-rolled instead of serde + bincode?
//!
//! The same reason the core has zero dependencies: the wire format *is*
//! curriculum. Every real device-adjacent protocol — ioctls, virtio
//! descriptors, GPU command packets themselves — is a hand-specified byte
//! layout, and the failure modes worth teaching (truncation, bad tags,
//! version skew, unbounded-length attacks) only become visible when you
//! own the bytes. The codecs are also *total*: any byte sequence decodes
//! to either a value or a `WireError`, never a panic — the daemon feeds
//! these functions bytes from untrusted sockets.
//!
//! All integers are little-endian (the byte order of every host this will
//! realistically run on; fixed explicitly so heterogeneous nodes agree).

use std::io::{self, Read, Write};

/// Protocol version, first byte of every payload. Bump on any layout
/// change; peers refuse mismatches loudly rather than misparse silently.
///
/// History: v1 = milestone 1 (opaque-cost kernels); v2 = milestone 2
/// (`KernelLaunch` carries threads/args/program, two new error variants);
/// v3 = milestone 3 (five migration messages: profile fetch, allocation
/// listing, dirty-page harvest, channel export/import); v4 = the
/// hardening pass (two new error variants, `ChannelExport` carries the
/// fence high-water mark); v5 = operability (QoS limits on profiles,
/// telemetry, QoS-window transfer for migration).
/// The command layout changed shape, so v1 peers must be refused — this
/// bump is the versioning policy doing its job, not an inconvenience.
pub const VERSION: u8 = 5;

/// Upper bound on a frame body. Guards the daemon against a malicious or
/// broken client sending a 4 GiB length prefix and OOMing the host — the
/// first rule of reading a length from a socket is to distrust it.
pub const MAX_FRAME_LEN: u32 = 16 * 1024 * 1024;

/// Everything that can go wrong turning bytes into values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WireError {
    /// Input ended before the value it promised.
    Truncated,
    /// A tag byte had no meaning at its position.
    BadTag {
        /// What was being decoded.
        context: &'static str,
        /// The offending byte.
        tag: u8,
    },
    /// A string field was not valid UTF-8.
    BadUtf8,
    /// Peer speaks a different protocol version.
    VersionMismatch {
        /// Version we implement.
        ours: u8,
        /// Version the peer sent.
        theirs: u8,
    },
    /// Frame length prefix exceeded `MAX_FRAME_LEN`.
    FrameTooLarge(u32),
    /// Frame decoded fine but had bytes left over — a sign the peer's
    /// encoder and our decoder disagree, which must never pass silently.
    TrailingBytes(usize),
}

impl std::fmt::Display for WireError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Truncated => write!(f, "message truncated"),
            Self::BadTag { context, tag } => write!(f, "bad tag {tag:#04x} decoding {context}"),
            Self::BadUtf8 => write!(f, "string field is not valid UTF-8"),
            Self::VersionMismatch { ours, theirs } => {
                write!(f, "protocol version mismatch: ours {ours}, theirs {theirs}")
            }
            Self::FrameTooLarge(n) => write!(f, "frame length {n} exceeds limit"),
            Self::TrailingBytes(n) => write!(f, "{n} trailing bytes after message"),
        }
    }
}

impl std::error::Error for WireError {}

/// Encoding cursor: append-only byte buffer with typed writers.
#[derive(Default)]
pub struct Enc {
    buf: Vec<u8>,
}

impl Enc {
    /// Fresh empty buffer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Finish, yielding the encoded bytes.
    pub fn into_bytes(self) -> Vec<u8> {
        self.buf
    }

    /// Append one byte.
    pub fn u8(&mut self, v: u8) {
        self.buf.push(v);
    }

    /// Append a little-endian u32.
    pub fn u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    /// Append a little-endian u64.
    pub fn u64(&mut self, v: u64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    /// Append a length-prefixed (u32) byte string.
    pub fn bytes(&mut self, v: &[u8]) {
        self.u32(v.len() as u32);
        self.buf.extend_from_slice(v);
    }

    /// Append a length-prefixed UTF-8 string.
    pub fn str(&mut self, v: &str) {
        self.bytes(v.as_bytes());
    }

    /// Append a bool as one byte (0/1).
    pub fn bool(&mut self, v: bool) {
        self.u8(v as u8);
    }
}

/// Decoding cursor over a received frame.
pub struct Dec<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Dec<'a> {
    /// Cursor at the start of `buf`.
    pub fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], WireError> {
        if self.buf.len() - self.pos < n {
            return Err(WireError::Truncated);
        }
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    /// Read one byte.
    pub fn u8(&mut self) -> Result<u8, WireError> {
        Ok(self.take(1)?[0])
    }

    /// Read a little-endian u32.
    pub fn u32(&mut self) -> Result<u32, WireError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().expect("len 4")))
    }

    /// Read a little-endian u64.
    pub fn u64(&mut self) -> Result<u64, WireError> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().expect("len 8")))
    }

    /// Read a length-prefixed byte string. The length is bounded by the
    /// remaining input, so a lying prefix yields `Truncated`, not a huge
    /// allocation.
    pub fn bytes(&mut self) -> Result<Vec<u8>, WireError> {
        let n = self.u32()? as usize;
        Ok(self.take(n)?.to_vec())
    }

    /// Read a length-prefixed UTF-8 string.
    pub fn str(&mut self) -> Result<String, WireError> {
        String::from_utf8(self.bytes()?).map_err(|_| WireError::BadUtf8)
    }

    /// Read a bool byte (anything nonzero would indicate encoder skew, so
    /// only 0/1 are accepted).
    pub fn bool(&mut self) -> Result<bool, WireError> {
        match self.u8()? {
            0 => Ok(false),
            1 => Ok(true),
            tag => Err(WireError::BadTag {
                context: "bool",
                tag,
            }),
        }
    }

    /// Assert the cursor consumed everything — call after decoding a full
    /// message so encoder/decoder skew is an error, not a mystery.
    pub fn finish(self) -> Result<(), WireError> {
        let left = self.buf.len() - self.pos;
        if left == 0 {
            Ok(())
        } else {
            Err(WireError::TrailingBytes(left))
        }
    }
}

// ---------------------------------------------------------------------------
// Framing
// ---------------------------------------------------------------------------

/// Write one frame: `[len: u32 LE][body]`. Flushes, because a frame is a
/// request/response boundary and sitting in a BufWriter would deadlock
/// both peers.
///
/// The size limit is enforced on the way *out*, not just on the way in.
/// An asymmetric limit is a protocol bug waiting to happen: a peer that
/// happily emits a 32 MiB frame its own reader would reject leaves the
/// stream unreadable and unrecoverable — the receiver cannot skip a body
/// it refused to size. Worse, `body.len() as u32` silently truncates
/// past 4 GiB, which would frame the *wrong number of bytes* and
/// desynchronize the connection permanently. Both are refused here, so
/// every frame this codebase writes is one it could also read.
pub fn write_frame(w: &mut impl Write, body: &[u8]) -> io::Result<()> {
    if body.len() > MAX_FRAME_LEN as usize {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "refusing to emit a {} B frame: limit is {MAX_FRAME_LEN} B",
                body.len()
            ),
        ));
    }
    let len = body.len() as u32;
    w.write_all(&len.to_le_bytes())?;
    w.write_all(body)?;
    w.flush()
}

/// Read one frame. `Ok(None)` means the peer closed cleanly *between*
/// frames (a normal hang-up); EOF mid-frame is an error (a torn message).
pub fn read_frame(r: &mut impl Read) -> io::Result<Option<Vec<u8>>> {
    let mut len_buf = [0u8; 4];
    // Hand-rolled "read exactly 4 or detect clean EOF at byte 0".
    let mut got = 0;
    while got < 4 {
        match r.read(&mut len_buf[got..])? {
            0 if got == 0 => return Ok(None), // clean close between frames
            0 => {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "connection closed mid-frame",
                ))
            }
            n => got += n,
        }
    }
    let len = u32::from_le_bytes(len_buf);
    if len > MAX_FRAME_LEN {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            WireError::FrameTooLarge(len).to_string(),
        ));
    }
    let mut body = vec![0u8; len as usize];
    r.read_exact(&mut body)?;
    Ok(Some(body))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_roundtrip() {
        let mut e = Enc::new();
        e.u8(7);
        e.u32(0xDEAD_BEEF);
        e.u64(u64::MAX);
        e.str("hello");
        e.bool(true);
        let bytes = e.into_bytes();

        let mut d = Dec::new(&bytes);
        assert_eq!(d.u8().unwrap(), 7);
        assert_eq!(d.u32().unwrap(), 0xDEAD_BEEF);
        assert_eq!(d.u64().unwrap(), u64::MAX);
        assert_eq!(d.str().unwrap(), "hello");
        assert!(d.bool().unwrap());
        d.finish().unwrap();
    }

    #[test]
    fn truncation_is_an_error_not_a_panic() {
        let mut e = Enc::new();
        e.u64(42);
        let bytes = e.into_bytes();
        let mut d = Dec::new(&bytes[..5]); // cut mid-integer
        assert_eq!(d.u64(), Err(WireError::Truncated));
    }

    #[test]
    fn lying_length_prefix_truncates_safely() {
        let mut e = Enc::new();
        e.u32(1_000_000); // claims a megabyte follows
        e.u8(1); // ...but only one byte does
        let bytes = e.into_bytes();
        let mut d = Dec::new(&bytes);
        assert_eq!(d.bytes(), Err(WireError::Truncated));
    }

    #[test]
    fn trailing_bytes_are_detected() {
        let mut e = Enc::new();
        e.u8(1);
        e.u8(2);
        let bytes = e.into_bytes();
        let mut d = Dec::new(&bytes);
        d.u8().unwrap();
        assert_eq!(d.finish(), Err(WireError::TrailingBytes(1)));
    }

    #[test]
    fn frame_roundtrip_and_clean_eof() {
        let mut pipe: Vec<u8> = Vec::new();
        write_frame(&mut pipe, b"abc").unwrap();
        write_frame(&mut pipe, b"").unwrap();
        let mut r = &pipe[..];
        assert_eq!(read_frame(&mut r).unwrap(), Some(b"abc".to_vec()));
        assert_eq!(read_frame(&mut r).unwrap(), Some(vec![]));
        assert_eq!(read_frame(&mut r).unwrap(), None); // clean EOF
    }

    #[test]
    fn oversized_frame_is_rejected_before_allocation() {
        let mut pipe = (MAX_FRAME_LEN + 1).to_le_bytes().to_vec();
        pipe.extend_from_slice(&[0; 8]);
        let mut r = &pipe[..];
        assert!(read_frame(&mut r).is_err());
    }

    #[test]
    fn torn_frame_is_an_error() {
        let mut pipe: Vec<u8> = Vec::new();
        write_frame(&mut pipe, b"abcdef").unwrap();
        let mut r = &pipe[..6]; // len prefix + 2 of 6 body bytes
        assert!(read_frame(&mut r).is_err());
    }
}
