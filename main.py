import numpy as np
from time import perf_counter
from tables import T2, Pi
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
    round_keys = [keys[4*i:i*4+4] for i in range(8)]
    return round_keys


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
    # l = block[0]
    # r = block[1]
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
        if len(block[i*8:i*8+8]) < 8:
            while len(block[i:i+8]) != 8:
                block += '0'
    return np.fromstring(block, np.uint8)


@cuda.jit()
def main_encrypt(encryption_keys, block, encrypt_block, table): # uint8
    c_table = cuda.shared.array((8, 16), dtype=numba.uint8)
    if cuda.threadIdx.x == 0:
        for i in range(8):
            for j in range(16):
                c_table[i, j] = table[i, j]
    cuda.syncthreads()
    for i in range(8*cuda.grid(1), len(encrypt_block), 8*cuda.gridsize(1)):
        encrypt(encryption_keys, block[i:i+8].view(numba.uint32),
                encrypt_block[i:i+8].view(numba.uint32), c_table)


@cuda.jit()
def main_decrypt(encryption_keys, encrypt_block, decrypt_block, table):
    c_table = cuda.shared.array((8, 16), dtype=numba.uint8)
    if cuda.threadIdx.x == 0:
        for i in range(8):
            for j in range(16):
                c_table[i, j] = table[i, j]
    cuda.syncthreads()
    for i in range(8*cuda.grid(1), len(encrypt_block), 8*cuda.gridsize(1)):
        decrypt(encryption_keys, encrypt_block[i:i+8].view(numba.uint32),
                decrypt_block[i:i+8].view(numba.uint32), c_table)


def main():
    # key = ('1'*4+'2'*4+'3'*4+'4'*4)*2
    key = np.array([204, 221, 238, 255, 136, 153, 170, 187, 68,  85, 102, 119, 0, 17, 34, 51, 243, 242, 241, 240, 247,
                    246, 245, 244, 251, 250, 249, 248, 255, 254, 253, 252], dtype=np.uint8)
    round_keys = iter_keys(key)
    encryption_keys = round_keys * 3 + round_keys[::-1]
    encryption_keys = np.array(encryption_keys, dtype=np.uint8)
    main_block = np.random.randint(0, 256, size=1024**2*64, dtype=np.uint8)
    # main_block = np.array([16, 50, 84, 118, 152, 186, 220, 254] * 131072, dtype=np.uint8)
    block = main_block
    # block = check_len_main_block(main_block)
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
    main_encrypt[128, 128](encryption_keys, block[:8], encrypt_block_cuda[:8], P_table)
    main_decrypt[128, 128](encryption_keys, encrypt_block_cuda[:8], decrypt_block_cuda[:8], P_table)
    print("Start encrypt")
    start = perf_counter()
    main_encrypt[128, 128](encryption_keys, block, encrypt_block_cuda, P_table)
    cuda.synchronize()
    end = perf_counter()
    encrypt_block = encrypt_block_cuda.copy_to_host()
    print(f'Encrypt_block: {encrypt_block[:8]}')
    # зашифрованный текст в приложении ГОСТа: [78, 233, 1, 229, 194, 216, 202, 61]
    print(f'Time encrypt: {end - start} sec')
    print("-" * 20)
    print("Start decrypt")
    start = perf_counter()
    main_decrypt[128, 128](encryption_keys, encrypt_block_cuda, decrypt_block_cuda, P_table)
    cuda.synchronize()
    print(f'Time decrypt: {perf_counter() - start} sec')
    decrypt_block = decrypt_block_cuda.copy_to_host()
    print(f'Decrypt_block: {decrypt_block[:8]}')
    # decrypt_str = [chr(decrypt_block[0])]
    # for i in range(1, len(main_block)):
    #     decrypt_str += chr(decrypt_block[i])
    # decrypt_str = ''.join(decrypt_str)
    if np.all(main_block == decrypt_block):
        print("Successful!")


if __name__ == '__main__':
    main()
