import numba
from numba import cuda
import numpy as np
from cpu import print_size
import time


def iter_keys(keys):
    round_keys = [keys[4 * i:i * 4 + 4] for i in range(8)]
    return round_keys


Pi = np.array([
    [12, 4, 6, 2, 10, 5, 11, 9, 14, 8, 13, 7, 0, 3, 15, 1],
    [6, 8, 2, 3, 9, 10, 5, 12, 1, 14, 4, 7, 11, 13, 0, 15],
    [11, 3, 5, 8, 2, 15, 10, 13, 14, 1, 7, 4, 12, 9, 6, 0],
    [12, 8, 2, 1, 13, 4, 15, 6, 7, 0, 10, 5, 3, 14, 9, 11],
    [7, 15, 5, 10, 8, 1, 6, 13, 0, 9, 3, 14, 11, 4, 2, 12],
    [5, 13, 15, 6, 9, 2, 12, 10, 11, 7, 8, 1, 4, 3, 14, 0],
    [8, 14, 2, 5, 6, 9, 1, 12, 15, 4, 11, 0, 13, 10, 3, 7],
    [1, 7, 14, 13, 0, 5, 8, 3, 4, 15, 10, 6, 9, 12, 11, 2]
], np.uint8)


@cuda.jit(numba.uint8(numba.int32, numba.uint8, numba.uint8[:, :]), device=True)
def T2(i: np.int32, val: np.uint8, P_table: np.ndarray):
    l = val & 0x0f
    h = (val & 0xf0) >> 4
    l = P_table[i * 2, l]
    h = P_table[i * 2 + 1, h]
    return (h << 4) | l


@cuda.jit(numba.uint8(numba.uint32, numba.uint32), device=True)
def get_byte(i, j):
    i = i >> (j * 8)
    return i & 0xff


@cuda.jit(numba.uint32(numba.uint32, numba.uint32, numba.uint8), device=True)
def set_byte(i, j, k):
    mask = 0xff << (j * 8)
    mask = ~mask
    i = i & mask
    v = numba.uint32(k)
    v = v << (j * 8)
    i = i | v
    return i


@cuda.jit(numba.uint32(numba.uint32, numba.uint8[:, :]), device=True)
def magma_T(in_data: np.uint32, table):
    for i in range(4):
        in_data = set_byte(in_data, i, T2(i, get_byte(in_data, i), table))
    return in_data


@cuda.jit(numba.uint32(numba.uint32, numba.uint32, numba.uint8[:, :]), device=True)
def magma_g(round_key, block, table):
    internal = round_key + block
    internal = magma_T(internal, table)
    internal = (internal << 11) | (internal >> 21)
    return internal


@cuda.jit(numba.void(numba.uint8[:], numba.uint32, numba.uint32, numba.uint8[:, :]), device=True)
def magma_G(round_key, l, r, table):
    t = r
    round_key = round_key.view(numba.uint32)
    t = magma_g(round_key[0], l, table)
    t = r ^ t
    r = l
    l = t
    return l, r


@cuda.jit(numba.void(numba.uint8[:], numba.uint32, numba.uint32, numba.uint8[:, :]), device=True)
def magma_G_Last(round_key, l, r, table):
    round_key = round_key.view(numba.uint32)
    t = magma_g(round_key[0], l, table)
    t = r ^ t
    r = t
    return l, r


@cuda.jit(numba.void(numba.uint8[:, :], numba.uint32[:], numba.uint32[:], numba.uint8[:, :]), device=True)
def encrypt(round_keys, block: np.uint32, out_block: np.uint32, table):
    l, r = magma_G(round_keys[0], block[0], block[1], table)
    for i in range(1, 31):
        l, r = magma_G(round_keys[i], l, r, table)
    l, r = magma_G_Last(round_keys[31], l, r, table)
    out_block[0] = l
    out_block[1] = r


@cuda.jit(numba.void(numba.uint8[:, :], numba.uint32[:], numba.uint32[:], numba.uint8[:, :]), device=True)
def decrypt(round_keys, block: np.uint8, out_block: np.uint8, table):
    l, r = magma_G(round_keys[31], block[0], block[1], table)
    for i in range(30, 0, -1):
        l, r = magma_G(round_keys[i], l, r, table)
    l, r = magma_G_Last(round_keys[0], l, r, table)
    out_block[0] = l
    out_block[1] = r


@numba.njit(parallel=True)
def check_len_main_block(block: str):
    for i in numba.prange(0, len(block)):
        if len(block[i * 8:i * 8 + 8]) < 8:
            while len(block[i:i + 8]) != 8:
                block += '0'
    return np.fromstring(block, np.uint8)


@cuda.jit()
def main_encrypt(encryption_keys, block, encrypt_block, table):  # uint8
    c_table = cuda.shared.array((8, 16), dtype=numba.uint8)
    if cuda.threadIdx.x == 0:
        for i in range(8):
            for j in range(16):
                c_table[i, j] = table[i, j]
    cuda.syncthreads()
    for i in range(8 * cuda.grid(1), len(encrypt_block), 8 * cuda.gridsize(1)):
        encrypt(encryption_keys, block[i:i + 8].view(numba.uint32),
                encrypt_block[i:i + 8].view(numba.uint32), c_table)


@cuda.jit()
def main_decrypt(encryption_keys, encrypt_block, decrypt_block, table):
    c_table = cuda.shared.array((8, 16), dtype=numba.uint8)
    if cuda.threadIdx.x == 0:
        for i in range(8):
            for j in range(16):
                c_table[i, j] = table[i, j]
    cuda.syncthreads()
    for i in range(8 * cuda.grid(1), len(encrypt_block), 8 * cuda.gridsize(1)):
        decrypt(encryption_keys, encrypt_block[i:i + 8].view(numba.uint32),
                decrypt_block[i:i + 8].view(numba.uint32), c_table)


def gpu(size: int):
    key = np.array([204, 221, 238, 255, 136, 153, 170, 187, 68, 85, 102, 119, 0, 17, 34, 51, 243, 242, 241, 240, 247,
                    246, 245, 244, 251, 250, 249, 248, 255, 254, 253, 252], dtype=np.uint8)
    round_keys = iter_keys(key)
    encryption_keys = round_keys * 3 + round_keys[::-1]
    encryption_keys = np.array(encryption_keys, dtype=np.uint8)
    main_block = np.random.randint(0, 256, size=size, dtype=np.uint8)
    block = main_block
    print_size(block.size)
    encrypt_block = block
    decrypt_block = block
    evtstart_enc = cuda.event(timing=True)
    evtend_enc = cuda.event(timing=True)
    evtstart_dec = cuda.event(timing=True)
    evtend_dec = cuda.event(timing=True)
    begin = time.time()
    P_table = cuda.to_device(Pi)
    encrypt_block_cuda = cuda.to_device(encrypt_block)
    block = cuda.to_device(block)
    encryption_keys = cuda.to_device(encryption_keys)
    decrypt_block_cuda = cuda.to_device(decrypt_block)
    main_encrypt[128, 128](encryption_keys, block[:8], encrypt_block_cuda[:8], P_table)
    main_decrypt[128, 128](encryption_keys, encrypt_block_cuda[:8], decrypt_block_cuda[:8], P_table)
    evtstart_enc.record()
    main_encrypt[128, 128](encryption_keys, block, encrypt_block_cuda, P_table)
    evtend_enc.record()
    evtend_enc.synchronize()
    time_ = float(cuda.event_elapsed_time(evtstart_enc, evtend_enc)) / 1000
    end = time.time()
    all_time = end - begin
    print(f'Time encrypting: {round(time_, 4)} sec')
    print(f'Time copying and encrypting: {round(all_time, 4)} sec')
    speed_encrypt = size / (time_ * 1024 ** 2)
    print(f'Speed encrypt: {round(speed_encrypt, 4)} Mb/sec')
    print("-" * 20)
    evtstart_dec.record()
    main_decrypt[128, 128](encryption_keys, encrypt_block_cuda, decrypt_block_cuda, P_table)
    evtend_dec.record()
    evtend_dec.synchronize()
    time_ = float(cuda.event_elapsed_time(evtstart_dec, evtend_dec)) / 1000
    print(f'Time decrypting: {round(time_, 4)} sec')
    speed_decrypt = size / (time_ * 1024 ** 2)
    print(f'Speed decrypting: {round(speed_decrypt, 4)} Mb/sec')
    if np.all(main_block == decrypt_block):
        return speed_encrypt, speed_decrypt
    return -1, -1
