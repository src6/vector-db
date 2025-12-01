use super::VectorStorage;
use crate::types::PointId;
use memmap2::{MmapMut, MmapOptions};
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::fs::{File, OpenOptions};
use std::io;
use std::path::{Path, PathBuf};

/// Fixed-dimension, preallocated mmap-backed vector storage.
#[derive(Debug)]
pub struct MmapStorage {
    path: PathBuf,
    dim: usize,
    capacity: usize,
    len: usize,
    _file: File, // keep file handle alive for the mmap duration
    mmap: MmapMut,
}

impl MmapStorage {
    /// Create or overwrite a mmap-backed store with a fixed capacity and dimension.
    pub fn create<P: AsRef<Path>>(path: P, dim: usize, capacity: usize) -> io::Result<Self> {
        let path_buf = path.as_ref().to_path_buf();
        let bytes = capacity
            .checked_mul(dim)
            .and_then(|v| v.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| io::Error::new(io::ErrorKind::Other, "capacity overflow"))?;
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .read(true)
            .truncate(true)
            .open(&path_buf)?;
        file.set_len(bytes as u64)?;
        let mmap = unsafe { MmapOptions::new().map_mut(&file)? };
        Ok(Self {
            path: path_buf,
            dim,
            capacity,
            len: 0,
            _file: file,
            mmap,
        })
    }

    fn write_vector(&mut self, offset: usize, vector: &[f32]) {
        let start = offset * self.dim;
        for (i, v) in vector.iter().enumerate() {
            let pos = (start + i) * std::mem::size_of::<f32>();
            self.mmap[pos..pos + 4].copy_from_slice(&v.to_le_bytes());
        }
    }

    fn read_vector(&self, idx: usize) -> Vec<f32> {
        let start = idx * self.dim * std::mem::size_of::<f32>();
        let mut out = Vec::with_capacity(self.dim);
        for i in 0..self.dim {
            let pos = start + i * std::mem::size_of::<f32>();
            let bytes: [u8; 4] = self.mmap[pos..pos + 4]
                .try_into()
                .expect("slice with incorrect length");
            out.push(f32::from_le_bytes(bytes));
        }
        out
    }
}

impl VectorStorage for MmapStorage {
    fn push(&mut self, vector: Vec<f32>) -> PointId {
        assert_eq!(vector.len(), self.dim, "dimension mismatch");
        if self.len >= self.capacity {
            panic!("mmap storage capacity exceeded");
        }
        let id = self.len;
        self.write_vector(id, &vector);
        self.len += 1;
        id
    }

    fn get(&self, id: PointId) -> Option<Cow<'_, [f32]>> {
        if id >= self.len {
            return None;
        }
        Some(Cow::Owned(self.read_vector(id)))
    }

    fn len(&self) -> usize {
        self.len
    }

    fn dim(&self) -> Option<usize> {
        Some(self.dim)
    }
}

impl Serialize for MmapStorage {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let state = (self.path.to_string_lossy(), self.dim, self.capacity, self.len);
        state.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for MmapStorage {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let (path_str, dim, capacity, len): (String, usize, usize, usize) =
            Deserialize::deserialize(deserializer)?;
        let path = PathBuf::from(path_str);
        let bytes = capacity
            .checked_mul(dim)
            .and_then(|v| v.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| serde::de::Error::custom("capacity overflow"))?;
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .map_err(serde::de::Error::custom)?;
        file.set_len(bytes as u64)
            .map_err(serde::de::Error::custom)?;
        let mmap = unsafe { MmapOptions::new().map_mut(&file) }
            .map_err(serde::de::Error::custom)?;
        Ok(Self {
            path,
            dim,
            capacity,
            len,
            _file: file,
            mmap,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::fs;

    #[test]
    fn push_and_get_round_trip() {
        let path = env::temp_dir().join("vector-db-mmap.bin");
        let _ = fs::remove_file(&path);
        let mut store = MmapStorage::create(&path, 3, 4).expect("create mmap");
        let id = store.push(vec![1.0, 2.0, 3.0]);
        assert_eq!(id, 0);
        let fetched = store.get(id).unwrap();
        assert_eq!(fetched.as_ref(), &[1.0, 2.0, 3.0]);
        fs::remove_file(&path).ok();
    }
}
