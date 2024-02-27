#!/usr/bin/env python3

import struct


def float_to_binary(num, bit_size=32):
    """Convert a float to its IEEE 754 binary representation."""
    if bit_size == 32:
        return format(struct.unpack("!I", struct.pack("!f", num))[0], "032b")
    elif bit_size == 64:
        return format(struct.unpack("!Q", struct.pack("!d", num))[0], "064b")


def binary_to_float(binary, bit_size=32):
    """Convert an IEEE 754 binary representation to a float."""
    if bit_size == 32:
        return struct.unpack("!f", struct.pack("!I", int(binary, 2)))[0]
    elif bit_size == 64:
        return struct.unpack("!d", struct.pack("!Q", int(binary, 2)))[0]


def print_float_range(lower_bound, upper_bound, bit_size=32):
    """Print all discrete IEEE float values between lower and upper bounds."""
    lower_binary = float_to_binary(lower_bound, bit_size)
    upper_binary = float_to_binary(upper_bound, bit_size)

    current = int(lower_binary, 2)
    end = int(upper_binary, 2)

    while current <= end:
        print(binary_to_float(format(current, "0" + str(bit_size) + "b"), bit_size))
        current += 1


# Example usage
# print_float_range(
#     12950820,
#     12950830,
#     32,
# )
print_float_range(
    161220,
    161230,
    32,
)
