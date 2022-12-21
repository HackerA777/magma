import numpy as np
import time
import numba


def iter_keys(keys):
    # keys = np.fromstring(keys, np.uint8)
    round_keys = [keys[4*i:i*4+4] for i in range(8)]
    return round_keys


def print_size(size):
    if size/1024 < 1.0:
        print(f'Size: {size}b')
    elif size/1024**2 < 1.0:
        print(f'Size: {size/1024} Kb')
    elif size/1024**3 < 1.0:
        print(f'Size: {size/1024**2} Mb')
    elif size/1024**4 < 1.0:
        print(f'Size: {size/1024**3}Gb')


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


@numba.njit(numba.uint8(numba.int32, numba.uint8), nogil=True, fastmath=True)
def T(i: np.int32, val: np.uint8):
    l = val & 0x0f
    h = (val & 0xf0) >> 4
    l = Pi[i*2, l]
    h = Pi[i*2+1, h]
    return (h << 4) | l


@numba.njit(numba.uint8(numba.uint32, numba.uint32), nogil=True)
def get_byte(i, j):
    i = i >> (j * 8)
    return i & 0xff


@numba.njit(numba.uint32(numba.uint32, numba.uint32, numba.uint8), nogil=True)
def set_byte(i, j, k):
    mask = 0xff << (j * 8)
    mask = ~mask
    i = i & mask
    v = numba.uint32(k)
    v = v << (j * 8)
    i = i | v
    return i


@numba.njit(numba.uint32(numba.uint32), nogil=True)
def magma_T(in_data: np.uint32):
    for i in range(4):
        in_data = set_byte(in_data, i, T(i, get_byte(in_data, i)))
    return in_data


@numba.njit(numba.uint32(numba.uint32, numba.uint32), nogil=True)
def magma_g(round_key, block):
    internal = round_key + block
    internal = magma_T(internal)
    internal = (internal << 11) | (internal >> 21)
    return internal


@numba.njit(numba.void(numba.uint8[:], numba.uint32[:], numba.uint32[:]), nogil=True)
def magma_G(round_key, block, out_block):
    l = block[0]
    r = block[1]
    t = r
    round_key = round_key.view(numba.uint32)
    t = magma_g(round_key[0], l)
    t = r ^ t
    r = l
    l = t
    out_block[0] = l
    out_block[1] = r


@numba.njit(numba.void(numba.uint8[:], numba.uint32[:], numba.uint32[:]), nogil=True)
def magma_G_Last(round_key, block, out_block):
    l = block[0]
    r = block[1]
    round_key = round_key.view(numba.uint32)
    t = magma_g(round_key[0], l)
    t = r ^ t
    r = t
    out_block[0] = l
    out_block[1] = r


@numba.njit(numba.void(numba.uint8[:, :], numba.uint32[:], numba.uint32[:]), nogil=True)
def encrypt(round_keys, block: np.uint32, out_block: np.uint32):
    magma_G(round_keys[0], block, out_block)
    for i in range(1, 31):
        magma_G(round_keys[i], out_block, out_block)
    magma_G_Last(round_keys[31], out_block, out_block)


@numba.njit(numba.void(numba.uint8[:, :], numba.uint32[:], numba.uint32[:]), nogil=True)
def decrypt(round_keys, block: np.uint8, out_block: np.uint8):
    magma_G(round_keys[31], block, out_block)
    for i in range(30, 0, -1):
        magma_G(round_keys[i], out_block, out_block)
    magma_G_Last(round_keys[0], out_block, out_block)


@numba.njit(numba.void(numba.uint8[:, :], numba.uint8[:], numba.uint8[:]), parallel=True, nogil=True)
def main_encrypt(encryption_keys, block, encrypt_block):
    for i in numba.prange(0, len(block)//8):
        encrypt(encryption_keys, block[i*8:i*8 + 8].view(numba.uint32), encrypt_block[i*8:i*8 + 8].view(numba.uint32))


@numba.njit(numba.void(numba.uint8[:, :], numba.uint8[:], numba.uint8[:]), parallel=True, nogil=True)
def main_decrypt(encryption_keys, encrypt_block, decrypt_block):
    for i in numba.prange(0, len(encrypt_block)//8):
        decrypt(encryption_keys, encrypt_block[i*8:i*8+8].view(numba.uint32),
                decrypt_block[i*8:i*8+8].view(numba.uint32))


def cpu(size: int):
    key = np.array([204, 221, 238, 255, 136, 153, 170, 187, 68,  85, 102, 119, 0, 17, 34, 51, 243, 242, 241, 240, 247,
                    246, 245, 244, 251, 250, 249, 248, 255, 254, 253, 252], dtype=np.uint8)
    main_block = np.random.randint(0, 256, size=size, dtype=np.uint8)
    round_keys = iter_keys(key)
    encryption_keys = round_keys * 3 + round_keys[::-1]
    encryption_keys = np.array(encryption_keys, dtype=np.uint8)
    block = main_block
    print_size(block.size)
    encrypt_block = block
    decrypt_block = block
    begin = time.time()
    main_encrypt(encryption_keys, block, encrypt_block)
    end = time.time()
    work_time_enc = end - begin
    print(f'Time encrypting: {round(work_time_enc, 4)} sec')
    speed_encrypt = size / (work_time_enc * 1024 ** 2)
    print(f'Speed encrypting: {round(speed_encrypt, 4)} Mb/sec')
    begin = time.time()
    main_decrypt(encryption_keys, encrypt_block, decrypt_block)
    end = time.time()
    work_time_dec = end - begin
    print(f'Time decrypting: {round(work_time_dec, 4)} sec')
    speed_decrypt = size / (work_time_dec * 1024 ** 2)
    print(f'Speed decrypting: {round(speed_decrypt, 4)} Mb/sec')
    if np.all(main_block == decrypt_block):
        return speed_encrypt, speed_decrypt, work_time_enc, work_time_dec
    return -1, -1, -1, -1
