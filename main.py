import numpy as np
from time import perf_counter
import numba
from tables import T


def print_size(size):
    if size/1024 < 1.0:
        print(f'Size: {size}b')
    elif size/1024**2 < 1.0:
        print(f'Size: {size/1024} Kb')
    elif size/1024**3 < 1.0:
        print(f'Size: {size/1024**2} Mb')
    elif size/1024**4 < 1.0:
        print(f'Size: {size/1024**3}Gb')


def iter_keys(keys):
    # keys = np.fromstring(keys, np.uint8)
    round_keys = [keys[4*i:i*4+4] for i in range(8)]
    return round_keys


@numba.njit(numba.uint8(numba.uint32, numba.uint32))
def get_byte(i, j):
    i = i >> (j * 8)
    return i & 0xff


@numba.njit(numba.uint32(numba.uint32, numba.uint32, numba.uint8))
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


@numba.njit(parallel=True)
def check_len_main_block(block: str):
    for i in numba.prange(0, len(block)):
        if len(block[i*8:i*8+8]) < 8:
            while len(block[i:i+8]) != 8:
                block += '0'
    return np.fromstring(block, np.uint8)


@numba.njit(numba.void(numba.uint8[:, :], numba.uint8[:], numba.uint8[:]), parallel=True, nogil=True)
def main_encrypt(encryption_keys, block, encrypt_block):
    for i in numba.prange(0, len(block)//8):
        encrypt(encryption_keys, block[i*8:i*8 + 8].view(numba.uint32), encrypt_block[i*8:i*8 + 8].view(numba.uint32))


@numba.njit(numba.void(numba.uint8[:, :], numba.uint8[:], numba.uint8[:]), parallel=True, nogil=True)
def main_decrypt(encryption_keys, encrypt_block, decrypt_block):
    for i in numba.prange(0, len(encrypt_block)//8):
        decrypt(encryption_keys, encrypt_block[i*8:i*8+8].view(numba.uint32), decrypt_block[i*8:i*8+8].view(numba.uint32))


def main():
    # key = ('1'*4+'2'*4+'3'*4+'4'*4)*2
    key = np.array([204, 221, 238, 255, 136, 153, 170, 187, 68,  85, 102, 119, 0, 17, 34, 51, 243, 242, 241, 240, 247,
                    246, 245, 244, 251, 250, 249, 248, 255, 254, 253, 252], dtype=np.uint8)
    main_block = np.random.randint(0, 256, size=1024**2*256, dtype=np.uint8)
    # main_block = np.array([16, 50, 84, 118, 152, 186, 220, 254] * 131072, dtype=np.uint8)
    round_keys = iter_keys(key)
    encryption_keys = round_keys * 3 + round_keys[::-1]
    encryption_keys = np.array(encryption_keys, dtype=np.uint8)
    # decryption_keys = encryption_keys[::-1]
    # block = check_len_main_block(main_block)
    block = main_block
    print_size(block.size)
    encrypt_block = block
    decrypt_block = block
    print(f'Block: {block}')
    print("Start encrypt")
    start = perf_counter()
    main_encrypt(encryption_keys, block, encrypt_block)
    end = perf_counter()
    print(f'Encrypt_block: {encrypt_block}')
    print(f'Time encrypt: {end - start} sec')
    print("Start decrypt")
    start = perf_counter()
    main_decrypt(encryption_keys, encrypt_block, decrypt_block)
    end = perf_counter()
    print(f'Time decrypt: {end - start} sec')
    print(f'Decrypt_block: {decrypt_block}')
    # decrypt_str = [chr(decrypt_block[0])]
    # for i in range(1, len(main_block)):
    #     decrypt_str += chr(decrypt_block[i])
    # decrypt_str = ''.join(decrypt_str)
    if np.all(main_block == decrypt_block):
        print("Successful!")


if __name__ == '__main__':
    main()
