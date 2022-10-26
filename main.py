import numpy as np
from time import perf_counter
from tables import T
import numba


def iter_keys(keys):
    # keys = np.fromstring(keys, np.uint8)
    round_keys = [keys[4*i:i*4+4] for i in range(8)]
    return round_keys


@numba.njit()
def add(a: np.uint8, b: np.uint8, c: np.uint8):
    c[...] = a ^ b


@numba.njit()
def add_32(a: np.uint8, b: np.uint8, c: np.uint8):
    a = a.view(np.uint32)
    b = b.view(np.uint32)
    c = c.view(np.uint32)
    c[...] = a + b


@numba.njit(numba.void(numba.uint8[:], numba.uint8[:]))
def magma_T(in_data: np.uint8, out_data: np.uint8):
    out_data[0] = T(0, in_data[0])
    out_data[1] = T(1, in_data[1])
    out_data[2] = T(2, in_data[2])
    out_data[3] = T(3, in_data[3])


@numba.njit(numba.void(numba.uint8[:], numba.uint8[:], numba.uint8[:]))
def magma_g(round_key, block, internal):
    add_32(block, round_key, internal)
    magma_T(internal, internal)
    out_data_32 = internal.view(np.uint32)
    out_data_32[0] = (out_data_32[0] << 11) | (out_data_32[0] >> 21)


@numba.njit(numba.void(numba.uint8[:], numba.uint8[:], numba.uint8[:], numba.uint8[:]))
def magma_G(round_key, block, out_block, t):
    l = block[:4]
    r = block[4:]
    t[...] = r
    magma_g(round_key, l, t)
    add(r, t, t)
    r[...] = l
    l[...] = t
    out_block[:4] = l
    out_block[4:] = r


@numba.njit(numba.void(numba.uint8[:], numba.uint8[:], numba.uint8[:], numba.uint8[:]))
def magma_G_Last(round_key, block, out_block, t):
    l = block[:4]
    r = block[4:]
    t[...] = r
    magma_g(round_key, l, t)
    add(r, t, t)
    r[...] = t
    out_block[:4] = l
    out_block[4:] = r


@numba.njit(numba.void(numba.uint8[:,:], numba.uint8[:], numba.uint8[:]))
def encrypt(round_keys, block: np.uint8, out_block: np.uint8):
    t = np.empty(4, np.uint8)
    magma_G(round_keys[0], block, out_block, t)
    for i in range(1, 31):
        magma_G(round_keys[i], out_block, out_block, t)
    magma_G_Last(round_keys[31], out_block, out_block, t)


@numba.njit(numba.void(numba.uint8[:,:], numba.uint8[:], numba.uint8[:]))
def decrypt(round_keys, block: np.uint8, out_block: np.uint8):
    t = np.empty(4, np.uint8)
    magma_G(round_keys[31], block, out_block,t)
    for i in range(30, 0, -1):
        magma_G(round_keys[i], out_block, out_block,t)
    magma_G_Last(round_keys[0], out_block, out_block,t)


def check_len_main_block(block: str):
    for i in range(0, len(block), 8):
        if len(block[i:i+8]) < 8:
            while len(block[i:i+8]) != 8:
                block += '0'
    return np.fromstring(block, np.uint8)


@numba.njit(numba.void(numba.uint8[:, :], numba.uint8[:], numba.uint8[:]))
def main_encrypt(encryption_keys, block, encrypt_block):
    for i in range(0, len(block), 8):
        encrypt(encryption_keys, block[i:i + 8], encrypt_block[i:i + 8])


@numba.njit(numba.void(numba.uint8[:, :], numba.uint8[:], numba.uint8[:]))
def main_decrypt(encryption_keys, encrypt_block, decrypt_block):
    for i in range(0, len(encrypt_block), 8):
        decrypt(encryption_keys, encrypt_block[i:i+8], decrypt_block[i:i+8])


def main():
    # key = ('1'*4+'2'*4+'3'*4+'4'*4)*2
    key = np.array([204, 221, 238, 255, 136, 153, 170, 187, 68,  85, 102, 119, 0, 17, 34, 51, 243, 242, 241, 240, 247,
                    246, 245, 244, 251, 250, 249, 248, 255, 254, 253, 252], dtype=np.uint8)
    # n = 55 000 000
    # main_block = 'abcdefgh' * 1
    main_block = "2TvºÜþ"
    round_keys = iter_keys(key)
    encryption_keys = round_keys * 3 + round_keys[::-1]
    encryption_keys = np.array(encryption_keys, dtype=np.uint8)
    # decryption_keys = encryption_keys[::-1]
    # block = check_len_main_block(main_block)
    block = np.array([16, 50, 84, 118, 152, 186, 220, 254]*131072, dtype=np.uint8)
    print(f'Size = {block.size/1024}Mb')
    encrypt_block = block
    decrypt_block = block
    print("Start encrypt")
    start = perf_counter()
    main_encrypt(encryption_keys, block, encrypt_block)
    end = perf_counter()
    print(f'Encrypt_block: {encrypt_block}')
    print(f'Time encrypt: {end - start}')
    start = perf_counter()
    main_decrypt(encryption_keys, encrypt_block, decrypt_block)
    end = perf_counter()
    print(f'Time decrypt: {end - start}')
    print(decrypt_block)
    decrypt_str = [chr(decrypt_block[0])]
    for i in range(1, len(main_block)):
        decrypt_str += chr(decrypt_block[i])
    decrypt_str = ''.join(decrypt_str)
    if main_block == decrypt_str:
        print("Successful!")


if __name__ == '__main__':
    main()
