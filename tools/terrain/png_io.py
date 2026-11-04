"""Minimal PNG writer (16-bit grayscale, 8-bit RGB) and hillshade preview.

Written by hand so the heightmap bit depth and colour space are exact; Blender's
image saver applies view transforms that can silently alter height values.
"""
import struct
import zlib

import numpy as np


def _chunk(tag, data):
    return (struct.pack(">I", len(data)) + tag + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))


def _write(path, raw_rows, width, height, bit_depth, color_type):
    ihdr = struct.pack(">IIBBBBB", width, height, bit_depth, color_type, 0, 0, 0)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(_chunk(b"IHDR", ihdr))
        f.write(_chunk(b"IDAT", zlib.compress(raw_rows, 9)))
        f.write(_chunk(b"IEND", b""))


def write_gray16(path, img_u16):
    """img_u16[row, col], row 0 = top of the image."""
    h, w = img_u16.shape
    be = img_u16.astype(">u2")
    rows = np.zeros((h, 1 + w * 2), np.uint8)
    rows[:, 1:] = be.view(np.uint8).reshape(h, w * 2)
    _write(path, rows.tobytes(), w, h, 16, 0)


def write_rgb8(path, img):
    h, w, _ = img.shape
    rows = np.zeros((h, 1 + w * 3), np.uint8)
    rows[:, 1:] = img.reshape(h, w * 3)
    _write(path, rows.tobytes(), w, h, 8, 2)


def write_rgba8(path, img):
    h, w, _ = img.shape
    rows = np.zeros((h, 1 + w * 4), np.uint8)
    rows[:, 1:] = img.reshape(h, w * 4)
    _write(path, rows.tobytes(), w, h, 8, 6)


def write_gray8(path, img):
    h, w = img.shape
    rows = np.zeros((h, 1 + w), np.uint8)
    rows[:, 1:] = img
    _write(path, rows.tobytes(), w, h, 8, 0)


def read_gray16(path):
    with open(path, "rb") as f:
        data = f.read()
    pos, idat, w, h = 8, b"", None, None
    while pos < len(data):
        n = struct.unpack(">I", data[pos:pos + 4])[0]
        tag = data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + n]
        if tag == b"IHDR":
            w, h = struct.unpack(">II", body[:8])
        elif tag == b"IDAT":
            idat += body
        pos += 12 + n
    raw = np.frombuffer(zlib.decompress(idat), np.uint8).reshape(h, 1 + w * 2)
    assert (raw[:, 0] == 0).all(), "only filter type 0 supported"
    return raw[:, 1:].copy().view(">u2").reshape(h, w).astype(np.uint16)


def heightmap_u16(h):
    """Normalise metres -> uint16, flipped so image top = north."""
    lo, hi = float(h.min()), float(h.max())
    u = np.round((h - lo) / (hi - lo) * 65535.0).astype(np.uint16)
    return u[::-1], lo, hi


def hillshade_rgb(h, cell, river_mask=None, azimuth=315.0, altitude=40.0):
    """Hypsometric tint x hillshade, image top = north. Used for fast QA."""
    gy, gx = np.gradient(h, cell)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    az, alt = np.radians(azimuth), np.radians(altitude)
    shade = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    shade = np.clip(shade, 0, 1)
    t = (h - h.min()) / (np.ptp(h) + 1e-9)
    stops = np.array([0.0, 0.08, 0.25, 0.5, 0.75, 1.0])
    cols = np.array([[96, 112, 70], [74, 104, 52], [58, 86, 44],
                     [92, 98, 70], [128, 120, 106], [214, 210, 204]], float)
    rgb = np.stack([np.interp(t, stops, cols[:, i]) for i in range(3)], -1)
    steep = np.clip((np.degrees(slope) - 30) / 15, 0, 1)[..., None]
    rgb = rgb * (1 - steep) + np.array([120, 114, 106]) * steep
    if river_mask is not None:
        rm = river_mask[..., None]
        rgb = rgb * (1 - rm) + np.array([60, 90, 120]) * rm
    rgb = rgb * (0.35 + 0.75 * shade[..., None])
    return np.clip(rgb, 0, 255).astype(np.uint8)[::-1]
