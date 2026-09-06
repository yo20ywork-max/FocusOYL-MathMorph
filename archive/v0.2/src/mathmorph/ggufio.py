"""Read GGUF v2/v3 and patch copies without rewriting unrelated bytes.

The core requires only NumPy. Optional gguf provides additional decoders.
No model code, pickle, network access, or external model repositories are used.
"""
from __future__ import annotations
import hashlib
import math
import os
from pathlib import Path
import struct
from dataclasses import dataclass
from typing import BinaryIO
import numpy as np

# ggml enum -> (name, elements per block, bytes per block).
TYPES = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2),
    2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22), 7: ("Q5_1", 32, 24),
    8: ("Q8_0", 32, 34), 9: ("Q8_1", 32, 40),
    10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144), 13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210), 15: ("Q8_K", 256, 292),
    16: ("IQ2_XXS", 256, 66), 17: ("IQ2_XS", 256, 74),
    18: ("IQ3_XXS", 256, 98), 19: ("IQ1_S", 256, 50),
    20: ("IQ4_NL", 32, 18), 21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82), 23: ("IQ4_XS", 256, 136),
    24: ("I8", 1, 1), 25: ("I16", 1, 2), 26: ("I32", 1, 4),
    27: ("I64", 1, 8), 28: ("F64", 1, 8), 29: ("IQ1_M", 256, 56),
    30: ("BF16", 1, 2), 34: ("TQ1_0", 256, 54), 35: ("TQ2_0", 256, 66),
    39: ("MXFP4", 32, 17),
}
SCALARS = {0:"B", 1:"b", 2:"H", 3:"h", 4:"I", 5:"i", 6:"f", 7:"B", 10:"Q", 11:"q", 12:"d"}

class GGUFError(ValueError):
    pass

@dataclass(frozen=True)
class Tensor:
    name: str
    dims: tuple[int, ...]  # GGML order, contiguous dimension first.
    qtype: int
    offset: int           # Absolute file offset.
    nbytes: int
    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(reversed(self.dims))
    @property
    def type_name(self) -> str:
        return TYPES.get(self.qtype, (f"UNKNOWN_{self.qtype}", 0, 0))[0]

class GGUF:
    """Bounded, read-only parser. Big endian and sharded input are rejected."""
    def __init__(self, path: str | Path):
        self.path = Path(path).resolve(strict=True)
        self.size = self.path.stat().st_size
        self.metadata: dict = {}
        self.tensors: dict[str, Tensor] = {}
        with self.path.open("rb") as f:
            if self._read(f, 4) != b"GGUF":
                raise GGUFError("Not a GGUF file")
            self.version = self._unpack(f, "I")
            if self.version not in (2, 3):
                raise GGUFError("Only little-endian GGUF v2/v3 is supported")
            nt, nk = self._unpack(f, "Q"), self._unpack(f, "Q")
            if nt > 1_000_000 or nk > 1_000_000:
                raise GGUFError("Unreasonable header counts")
            for _ in range(nk):
                key = self._string(f, 65535)
                if key in self.metadata:
                    raise GGUFError(f"Duplicate metadata key: {key}")
                typ = self._unpack(f, "I")
                self.metadata[key] = self._value(f, typ)
            self.kv_end = f.tell()
            self.kv_count = nk
            self.alignment = self.metadata.get("general.alignment", 32)
            if not isinstance(self.alignment, int) or self.alignment < 1 or self.alignment > 1048576 or self.alignment & (self.alignment-1):
                raise GGUFError("Invalid alignment")
            if int(self.metadata.get("split.count", 1)) != 1:
                raise GGUFError("Sharded files must be merged before conversion")
            infos = []
            names = set()
            for _ in range(nt):
                name = self._string(f, 4096)
                if name in names:
                    raise GGUFError(f"Duplicate tensor: {name}")
                names.add(name)
                nd = self._unpack(f, "I")
                if not 1 <= nd <= 4:
                    raise GGUFError("Tensor dimensionality outside supported range")
                dims = tuple(self._unpack(f, "Q") for _ in range(nd))
                if any(x < 1 for x in dims):
                    raise GGUFError("Empty tensor")
                qt, rel = self._unpack(f, "I"), self._unpack(f, "Q")
                if rel % self.alignment:
                    raise GGUFError("Unaligned tensor")
                infos.append((name, dims, qt, rel))
            self.data_start = self._align(f.tell())
            sorted_infos = sorted(infos, key=lambda item: item[3])
            for i, (name, dims, qt, rel) in enumerate(sorted_infos):
                off = self.data_start + rel
                end = self.data_start + sorted_infos[i+1][3] if i+1 < len(sorted_infos) else self.size
                if qt in TYPES:
                    _, block, nb = TYPES[qt]
                    if dims[0] % block:
                        raise GGUFError(f"Invalid quantization row size: {name}")
                    nbytes = math.prod(dims) // block * nb
                else:
                    # Unknown types are inspectable and copied, never modified.
                    nbytes = end-off
                if off < self.data_start or nbytes <= 0 or off+nbytes > end or end > self.size:
                    raise GGUFError(f"Truncated or overlapping tensor: {name}")
                self.tensors[name] = Tensor(name, dims, qt, off, nbytes)
            if self.data_start > self.size:
                raise GGUFError("Truncated data section")

    def _read(self, f: BinaryIO, n: int) -> bytes:
        if n < 0 or f.tell()+n > self.size:
            raise GGUFError("Truncated GGUF")
        b = f.read(n)
        if len(b) != n:
            raise GGUFError("Unexpected EOF")
        return b
    def _unpack(self, f, fmt):
        return struct.unpack("<"+fmt, self._read(f, struct.calcsize("<"+fmt)))[0]
    def _string(self, f, limit=64*1024*1024):
        n = self._unpack(f, "Q")
        if n > limit:
            raise GGUFError("String exceeds safety limit")
        return self._read(f, n).decode("utf-8")
    def _value(self, f, typ, depth=0):
        if depth > 8:
            raise GGUFError("Metadata nesting exceeds safety limit")
        if typ in SCALARS:
            x = self._unpack(f, SCALARS[typ])
            if typ == 7 and x not in (0, 1):
                raise GGUFError("Invalid boolean")
            return bool(x) if typ == 7 else x
        if typ == 8:
            return self._string(f)
        if typ == 9:
            subtype, n = self._unpack(f,"I"), self._unpack(f,"Q")
            if n > 10_000_000:
                raise GGUFError("Metadata array exceeds safety limit")
            # Skip large tokenizer arrays; preserve exact bytes in every output.
            if subtype in SCALARS:
                size = struct.calcsize("<"+SCALARS[subtype]) * n
                if f.tell()+size > self.size:
                    raise GGUFError("Truncated metadata array")
                f.seek(size, 1)
            else:
                for _ in range(n):
                    self._value(f, subtype, depth+1)
            return {"array_type":subtype, "length":n, "preserved_as_raw_bytes":True}
        raise GGUFError(f"Unsupported metadata value type {typ}")
    def _align(self, n):
        return (n+self.alignment-1)//self.alignment*self.alignment
    def raw(self, name: str) -> bytes:
        t = self.tensors[name]
        with self.path.open("rb") as f:
            f.seek(t.offset)
            return self._read(f, t.nbytes)
    def matrix(self, name: str) -> np.ndarray:
        t = self.tensors[name]
        return decode(self.raw(name), t.qtype, t.shape)
    def write_single_f32(self, path: Path, name: str, array: np.ndarray):
        """Scratch input for an explicitly supplied llama-quantize executable."""
        a = np.ascontiguousarray(array, dtype="<f4")
        with self.path.open("rb") as src, Path(path).open("xb") as f:
            f.write(b"GGUF"+struct.pack("<IQQ", self.version, 1, self.kv_count))
            src.seek(24)
            f.write(src.read(self.kv_end-24))
            nb = name.encode("utf-8")
            f.write(struct.pack("<Q",len(nb))+nb+struct.pack("<I",a.ndim))
            for dim in reversed(a.shape):
                f.write(struct.pack("<Q",dim))
            f.write(struct.pack("<IQ",0,0))
            f.write(b"\0"*(self._align(f.tell())-f.tell()))
            f.write(a.tobytes())


def decode(raw: bytes, qtype: int, shape: tuple[int,...]) -> np.ndarray:
    if qtype in (0,1):
        return np.frombuffer(raw,dtype="<f4" if qtype==0 else "<f2").astype(np.float32).reshape(shape)
    if qtype==30:
        return (np.frombuffer(raw,dtype="<u2").astype(np.uint32)<<16).view(np.float32).reshape(shape)
    if qtype in (2,3,8):
        nb = TYPES[qtype][2]
        b = np.frombuffer(raw,dtype=np.uint8).reshape(-1,nb)
        d = b[:,:2].copy().view("<f2").astype(np.float32)
        if qtype==8:
            x = b[:,2:].copy().view(np.int8).astype(np.float32)*d
        else:
            q = b[:,2 if qtype==2 else 4:]
            x = np.concatenate((q&15,q>>4),axis=1).astype(np.float32)
            if qtype==2:
                x = (x-8)*d
            else:
                m = b[:,2:4].copy().view("<f2").astype(np.float32)
                x = x*d+m
        return x.reshape(shape)
    try:
        import gguf
        t = gguf.GGMLQuantizationType(qtype)
        block, nb = gguf.GGML_QUANT_SIZES[t]
        bshape = (*shape[:-1],shape[-1]//block*nb)
        return np.asarray(gguf.quants.dequantize(np.frombuffer(raw,dtype=np.uint8).reshape(bshape),t),dtype=np.float32).reshape(shape)
    except (ImportError, ValueError, KeyError, AttributeError, NotImplementedError) as exc:
        raise GGUFError(f"Decoder unavailable for type {qtype}; install compatible gguf") from exc


def encode_native(a: np.ndarray, qtype: int) -> bytes:
    a = np.ascontiguousarray(a,dtype=np.float32)
    if not np.isfinite(a).all():
        raise GGUFError("Cannot encode non-finite tensor")
    if qtype in (0,1):
        with np.errstate(over="ignore"):
            out = a.astype("<f4" if qtype==0 else "<f2")
        if not np.isfinite(out).all():
            raise GGUFError("FP16 overflow")
        return out.tobytes()
    if qtype==30:
        u=a.view(np.uint32).astype(np.uint64)
        return ((u+0x7fff+((u>>16)&1))>>16).astype("<u2").tobytes()
    raise GGUFError("Quantized writes require --quantizer; no guessed encoder is used")


def sha256(path: str | Path) -> str:
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""):
            h.update(b)
    return h.hexdigest()


def assert_only_patches(source: Path, candidate: Path, patches: dict[str, bytes]) -> dict:
    """Full-file byte comparison except precisely declared tensor spans."""
    original, changed = GGUF(source), GGUF(candidate)
    if original.size != changed.size or original.tensors != changed.tensors:
        raise GGUFError("Tensor layout changed")
    ranges = sorted((original.tensors[k].offset,original.tensors[k].nbytes,k) for k in patches)
    checked=0
    with Path(source).open("rb") as a, Path(candidate).open("rb") as b:
        pos=0
        for off,n,name in ranges+[(original.size,0,"")]:
            while pos < off:
                count=min(off-pos,8*1024*1024)
                if a.read(count)!=b.read(count):
                    raise GGUFError(f"Undeclared byte change at or after {pos}")
                checked+=count
                pos+=count
            if n:
                a.seek(n,1)
                if b.read(n)!=patches[name]:
                    raise GGUFError(f"Patch mismatch: {name}")
                pos+=n
    return {"unaltered_bytes_verified":checked,"file_size_unchanged":True,"metadata_byte_identical":True,"tensor_layout_unchanged":True}
