import tables


def sum_mod_2(block_a, block_b, block_c):
    for i in range(len(block_a)):
        block_c[i] = block_a[i] ^ block_b[i]


def sum_mod_32(block_a, block_b, bloc_c):
    internal = 0
    for i in range(0, len(block_a)):
        i = i * (-1) - 1
        internal = block_a[i] + block_b[i] + (internal >> 8)
        bloc_c[i] = internal


import array

import random
import string

char = 'abcdefgh'

block_a_1 = bytearray()
block_a_0 = bytearray()
block_b_1 = bytearray()
block_b_0 = bytearray()

for i in range(int(len(char)/2)):
    block_a_1 += char[i].encode('utf16')
    block_a_0 += char[-i-1].encode('utf16')
block_a_0 = block_a_0[::-1]
print("block_a_1: ", bytearray(block_a_1))
print("block_a_0: ", bytearray(block_a_0))
block_b_1 = block_a_0

# keys = random.choices(string.ascii_letters, k=26) + random.choices(string.digits, k=6)
# random.shuffle(keys)
keys = ['J', 'r', 'i', 'g', 'I', 'C', 'Q', 'o', '9', 'R', 'S', 'q', 'R', 'm', 'W', '6', '6', 'k', 'X', 'L', 't', 'G', 'Q', 'A', '6', 'D', '6', 'g', 'v', '5', 'P', 'Y']
print(f'Keys: {keys}')
key = bytearray()
for k in keys:
    k = str(k)
    key += k.encode('utf16')
key = bytearray(key)
print(f'Key: {type(key)}')
result = key.decode('utf16')
print(f'result: {result}')
iter_key = bytearray()
# for i in range(32):
#     iter_key += str(i).encode('utf16')
iter_key = [str(i).encode('utf16') for i in range(32)]
print(f'iter_key: {type(iter_key)}')
iter_key[7] = key[0:16]
iter_key[6] = key[0:16+16]
iter_key[5] = key[0:16+32]
iter_key[4] = key[0:16+48]
iter_key[3] = key[0:16+64]
iter_key[2] = key[0:16+80]
iter_key[1] = key[0:16+96]
iter_key[0] = key[0:16+112]
print(iter_key[0].decode('utf16'))
