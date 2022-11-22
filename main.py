import numpy as np
from time import perf_counter
from tables import T2, Pi, T
import numba
from numba import cuda


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


@cuda.jit(numba.void(numba.uint8[:], numba.uint8[:], numba.uint8[:]), device=True)
# @numba.njit(numba.void(numba.uint8[:], numba.uint8[:], numba.uint8[:]), nogil=True, fastmath=True)
def xor(a: np.uint8, b: np.uint8, c: np.uint8):
    a = a.view(np.uint32)
    b = b.view(np.uint32)
    c = c.view(np.uint32)
    c[0] = a[0] ^ b[0]


@cuda.jit(numba.void(numba.uint8[:], numba.uint8[:], numba.uint8[:]), device=True)
# @numba.njit(numba.void(numba.uint8[:], numba.uint8[:], numba.uint8[:]), nogil=True, fastmath=True)
def add_32(a: np.uint8, b: np.uint8, c: np.uint8):
    a = a.view(np.uint32)
    b = b.view(np.uint32)
    c = c.view(np.uint32)
    c[0] = a[0] + b[0]


@cuda.jit(numba.void(numba.uint8[:], numba.uint8[:], numba.uint8[:, :]), device=True)
# @numba.njit(numba.void(numba.uint8[:], numba.uint8[:]), nogil=True)
def magma_T(in_data: np.uint8, out_data: np.uint8, table: np.array):
    out_data[0] = T2(0, in_data[0], table)
    out_data[1] = T2(1, in_data[1], table)
    out_data[2] = T2(2, in_data[2], table)
    out_data[3] = T2(3, in_data[3], table)


@cuda.jit(numba.void(numba.uint8[:], numba.uint8[:], numba.uint8[:], numba.uint8[:, :]), device=True)
# @numba.njit(numba.void(numba.uint8[:], numba.uint8[:], numba.uint8[:]), nogil=True)
def magma_g(round_key, block, internal, table):
    add_32(block, round_key, internal)
    magma_T(internal, internal, table)
    out_data_32 = internal.view(np.uint32)
    out_data_32[0] = (out_data_32[0] << 11) | (out_data_32[0] >> 21)


@cuda.jit(numba.void(numba.uint8[:], numba.uint8[:], numba.uint8[:], numba.uint8[:], numba.uint8[:, :]), device=True)
# @numba.njit(numba.void(numba.uint8[:], numba.uint8[:], numba.uint8[:], numba.uint8[:]), nogil=True)
def magma_G(round_key, block, out_block, t, table):
    l = block[:4]
    r = block[4:]
    for i in range(4):
        t[i] = r[i]
    # t[:4] = r[:4]
    # t = r
    magma_g(round_key, l, t, table)
    xor(r, t, t)
    for i in range(4):
        t[i] = l[i]
    # r[0:4] = l[:4]
    # r = l
    for i in range(4):
        l[i] = t[i]
    # l[...] = t
    # l = t
    for i in range(4):
        out_block[i] = l[i]
        out_block[i+4] = r[i]
    # out_block[:4] = l
    # out_block[4:] = r


@cuda.jit(numba.void(numba.uint8[:], numba.uint8[:], numba.uint8[:], numba.uint8[:], numba.uint8[:, :]), device=True)
# @numba.njit(numba.void(numba.uint8[:], numba.uint8[:], numba.uint8[:], numba.uint8[:]), nogil=True)
def magma_G_Last(round_key, block, out_block, t, table):
    l = block[:4]
    r = block[4:]
    for i in range(4):
        t[i] = r[i]
    # t[...] = r
    # t = r
    magma_g(round_key, l, t, table)
    xor(r, t, t)
    for i in range(4):
        r[i] = t[i]
    # r[...] = t
    # r = t
    for i in range(4):
        out_block[i] = l[i]
        out_block[i+4] = r[i]
    # out_block[:4] = l
    # out_block[4:] = r


@cuda.jit(numba.void(numba.uint8[:, :], numba.uint8[:], numba.uint8[:], numba.uint8[:, :]), device=True)
# @numba.njit(numba.void(numba.uint8[:, :], numba.uint8[:], numba.uint8[:]), nogil=True)
def encrypt(round_keys: np.array, block: np.uint8, out_block: np.uint8, table: np.array):
    t = cuda.shared.array(shape=8, dtype=numba.uint8)
    magma_G(round_keys[0], block, out_block, t, table)
    for i in range(1, 31):
        magma_G(round_keys[i], out_block, out_block, t, table)
    magma_G_Last(round_keys[31], out_block, out_block, t, table)


@cuda.jit(numba.void(numba.uint8[:, :], numba.uint8[:], numba.uint8[:], numba.uint8[:, :]), device=True)
# @numba.njit(numba.void(numba.uint8[:, :], numba.uint8[:], numba.uint8[:]), nogil=True)
def decrypt(round_keys: np.array, block: np.uint8, out_block: np.uint8, table: np.array):
    # t = np.empty(4, np.uint8)
    t = cuda.shared.array(shape=8, dtype=numba.uint8)
    magma_G(round_keys[31], block, out_block, t, table)
    for i in range(30, 0, -1):
        magma_G(round_keys[i], out_block, out_block, t, table)
    magma_G_Last(round_keys[0], out_block, out_block, t, table)


@numba.njit(parallel=True)
def check_len_main_block(block: str):
    for i in numba.prange(0, len(block)):
        if len(block[i*8:i*8+8]) < 8:
            while len(block[i:i+8]) != 8:
                block += '0'
    return np.fromstring(block, np.uint8)


@cuda.jit()
# @numba.njit(numba.void(numba.uint8[:, :], numba.uint8[:], numba.uint8[:]), parallel=True, nogil=True)
def main_encrypt(encryption_keys, block, encrypt_block, table):
    # for i in numba.prange(0, len(block)//8):
    #     encrypt(encryption_keys, block[i*8:i*8 + 8], encrypt_block[i*8:i*8 + 8])
    # t = block[:8].copy()
    # t = cuda.shared.array(shape=8, dtype=numba.uint8)
    for i in range(8*cuda.grid(1), len(block), 8*cuda.gridsize(1)):
        encrypt(encryption_keys, block[i:i+8], encrypt_block[i:i+8], table)


@cuda.jit()
# @numba.njit(numba.void(numba.uint8[:, :], numba.uint8[:], numba.uint8[:]), parallel=True, nogil=True)
def main_decrypt(encryption_keys, encrypt_block, decrypt_block, table):
    # for i in numba.prange(0, len(encrypt_block)//8):
    #     decrypt(encryption_keys, encrypt_block[i*8:i*8+8], decrypt_block[i*8:i*8+8])
    # t = encrypt_block[:8].copy()
    # t = cuda.shared.array(shape=8, dtype=numba.uint8)
    for i in range(8*cuda.grid(1), len(encrypt_block), 8*cuda.gridsize(1)):
        decrypt(encryption_keys, encrypt_block[i:i+8], decrypt_block[i:i+8], table)


def main():
    # key = ('1'*4+'2'*4+'3'*4+'4'*4)*2
    key = np.array([204, 221, 238, 255, 136, 153, 170, 187, 68,  85, 102, 119, 0, 17, 34, 51, 243, 242, 241, 240, 247,
                    246, 245, 244, 251, 250, 249, 248, 255, 254, 253, 252], dtype=np.uint8)
    # main_block = np.random.randint(0, 250, size=1048576, dtype=np.uint8)
    round_keys = iter_keys(key)
    encryption_keys = round_keys * 3 + round_keys[::-1]
    encryption_keys = np.array(encryption_keys, dtype=np.uint8)
    # decryption_keys = encryption_keys[::-1]
    # block = check_len_main_block(main_block)
    # block = main_block
    block = np.array([16, 50, 84, 118, 152, 186, 220, 254] * 81920, dtype=np.uint8)
    main_block = block
    print_size(block.size)
    encrypt_block = block
    decrypt_block = block
    print(f'Block: {block[:8]}')
    print("-"*20)
    P_table = cuda.to_device(Pi)
    encrypt_block_cuda = cuda.to_device(encrypt_block)
    block = cuda.to_device(block)
    encryption_keys = cuda.to_device(encryption_keys)
    decrypt_block_cuda = cuda.to_device(decrypt_block)
    print("-"*20)
    print("Start encrypt")
    start = perf_counter()
    main_encrypt[128, 100](encryption_keys, block, encrypt_block_cuda, P_table)
    cuda.synchronize()
    end = perf_counter()
    # main_encrypt.parallel_diagnostics(level=4)
    encrypt_block = encrypt_block_cuda.copy_to_host()
    print(f'Encrypt_block: {encrypt_block[:8]}')
    # зашифрованный текст в приложении ГОСТа: [78, 233, 1, 229, 194, 216, 202, 61]
    print(f'Time encrypt: {end - start} sec')
    print("Start decrypt")
    start = perf_counter()
    main_decrypt[128, 100](encryption_keys, encrypt_block_cuda, decrypt_block_cuda, P_table)
    cuda.synchronize()
    print(f'Time decrypt: {perf_counter() - start} sec')
    decrypt_block = decrypt_block_cuda.copy_to_host()
    print(f'Decrypt_block: {decrypt_block[:8]}')
    # decrypt_str = [chr(decrypt_block[0])]
    # for i in range(1, len(main_block)):
    #     decrypt_str += chr(decrypt_block[i])
    # decrypt_str = ''.join(decrypt_str)
    if main_block.all() == decrypt_block.all():
        print("Successful!")


if __name__ == '__main__':
    main()
