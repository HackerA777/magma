import numpy as np
from time import perf_counter
from tables import Pi


def iter_keys(keys):
    keys = np.fromstring(keys, np.uint8)
    round_keys = [keys[4*i:i*4+4] for i in range(8)]
    return round_keys


def add(a: np.uint8, b: np.uint8, c: np.uint8):
    for i in range(4):
        c[i] = a[i] ^ b[i]


def add_32(a: np.uint8, b: np.uint8, c: np.uint8):
    internal = np.uint8(0)
    # internal = internal.view(np.uint32)
    for i in range(4):
        internal = a[i] + b[i] + (internal >> 8)
        c[i] = internal & 0xFF


def magma_T(in_data: np.uint8, out_data: np.uint8):
    for i in range(4):
        first_part_byte = (in_data[i] & 0x0F)
        second_part_byte = (in_data[i] & 0xF0) >> 4
        first_part_byte = Pi[i * 2][first_part_byte]
        second_part_byte = Pi[i * 2 + 1][second_part_byte]
        out_data[i] = (first_part_byte << 4) | second_part_byte


def magma_g(round_key, block, out_data):
    internal = np.array([0, 0, 0, 0], dtype=np.uint8)
    # out_data_32 = np.uint32(0)
    add_32(block, round_key, internal)
    magma_T(internal, internal)
    out_data_32 = internal[3]
    out_data_32 = (out_data_32 << 8) + internal[2]
    out_data_32 = (out_data_32 << 8) + internal[1]
    out_data_32 = (out_data_32 << 8) + internal[0]
    out_data_32 = (out_data_32 << 11) | (out_data_32 >> 21)
    out_data[0] = out_data_32
    out_data[1] = out_data_32 >> 8
    out_data[2] = out_data_32 >> 16
    out_data[3] = out_data_32 >> 24


def magma_G(round_key, block, out_block):
    l = block[:4]
    r = block[4:]

    t = np.copy(r).view(np.uint8)
    magma_g(round_key, l, t)
    add(r, t, t)
    for i in range(4):
        r[i] = l[i]
        l[i] = t[i]
    for i in range(4):
        out_block[i] = l[i]
        out_block[4+i] = r[i]


def magma_G_Last(round_key, block, out_block):
    l = block[:4]
    r = block[4:]

    t = np.copy(r).view(np.uint8)
    magma_g(round_key, l, t)
    add(r, t, t)
    for i in range(4):
        r[i] = t[i]
    for i in range(4):
        out_block[i] = l[i]
        out_block[4 + i] = r[i]


def encrypt(round_keys, block: np.uint8, out_block: np.uint8):
    magma_G(round_keys[0], block, out_block)
    for i in range(1, 31):
        magma_G(round_keys[i], out_block, out_block)
    magma_G_Last(round_keys[31], out_block, out_block)


def decrypt(round_keys, block: np.uint8, out_block: np.uint8):
    magma_G(round_keys[31], block, out_block)
    for i in range(30, 0, -1):
        magma_G(round_keys[i], out_block, out_block)
    magma_G_Last(round_keys[0], out_block, out_block)


def check_len_main_block(block: str):
    for i in range(0, len(block), 8):
        # print(f'block: {block[i:i+8]}')
        if len(block[i:i+8]) < 8:
            while(len(block[i:i+8]) != 8):
                block += '0'
                # print(block[i:i+8])
    block = np.fromstring(block, np.uint8)
    return block


def main():
    key = ('1'*4+'2'*4+'3'*4+'4'*4)*2
    # n = 55 000 000
    # main_block = ('1'*4+'2'*4+'3'*4+'4'*4+'5'*4) * 1
    main_block = 'abcdefgh' * 225000
    # block = np.fromstring(block, np.uint8)
    round_keys = iter_keys(key)
    encryption_keys = round_keys * 3 + round_keys[::-1]
    decryption_keys = encryption_keys[::-1]
    block = check_len_main_block(main_block)
    print(block)
    print(f'Size: {block.size}')
    encrypt_block = block
    decrypt_block = block
    print("Start encrypt")
    start = perf_counter()
    for i in range(0, len(block), 8):
        encrypt(encryption_keys, block[i:i+8], encrypt_block[i:i+8])
    print(f'encrypt: {encrypt_block}')
    # encrypt(decryption_keys, encrypt_block, decrypt_block)
    # print(f'decrypt: {decrypt_block}')
        # decrypt(encryption_keys, encrypt_block[i:i+8], decrypt_block[i:i+8])
        # print(f'decrypt: {decrypt_block}')
    end = perf_counter()
    print(f'Time encrypt: {end - start}')
    start = perf_counter()
    for i in range(0, len(block), 8):
        decrypt(encryption_keys, encrypt_block[i:i+8], decrypt_block[i:i+8])
    decrypt_str = [chr(decrypt_block[0])]
    end = perf_counter()
    print(f'Time decrypt: {end - start}')
    print(decrypt_block)
    for i in range(1, len(main_block)):
        decrypt_str += chr(decrypt_block[i])
    decrypt_str = ''.join(decrypt_str)
    if main_block == decrypt_str:
        print("Done!")


if __name__ == '__main__':
    main()
