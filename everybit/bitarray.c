/**
 * Copyright (c) 2012 MIT License by 6.172 Staff
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 **/

// Implements the ADT specified in bitarray.h as a packed array of bits; a bit
// array containing bit_sz bits will consume roughly bit_sz/8 bytes of
// memory.


#include "./bitarray.h"

#include <assert.h>
#include <immintrin.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <sys/mman.h>
#include <sys/types.h>


// ********************************* Types **********************************

// Concrete data type representing an array of bits.
struct bitarray {
    // The number of bits represented by this bit array.
    // Need not be divisible by 8.
    size_t bit_sz;

    // The underlying memory buffer that stores the bits in
    // packed form (8 per byte).
    char *buf;
};

typedef __m512i pocket;
static constexpr unsigned pocket_bit_size = sizeof(pocket) * CHAR_BIT;
static constexpr unsigned word_bit_size = sizeof(uint64_t) * CHAR_BIT;
static_assert(sizeof(pocket) % sizeof(uint64_t) == 0);
static constexpr unsigned words_in_pocket = sizeof(pocket) / sizeof(uint64_t);


// ******************** Prototypes for static functions *********************

// Rotates a subarray left by an arbitrary number of bits.
//
// bit_offset is the index of the start of the subarray
// bit_length is the length of the subarray, in bits
// bit_left_amount is the number of places to rotate the
//                    subarray left
//
// The subarray spans the half-open interval
// [bit_offset, bit_offset + bit_length)
// That is, the start is inclusive, but the end is exclusive.
static void bitarray_rotate_left(bitarray_t *const bitarray,
                                 const size_t bit_offset,
                                 const size_t bit_length,
                                 const size_t bit_left_amount);

// Portable modulo operation that supports negative dividends.
//
// Many programming languages define modulo in a manner incompatible with its
// widely-accepted mathematical definition.
// http://stackoverflow.com/questions/1907565/c-python-different-behaviour-of-the-modulo-operation
// provides details; in particular, C's modulo
// operator (which the standard calls a "remainder" operator) yields a result
// signed identically to the dividend e.g., -1 % 10 yields -1.
// This is obviously unacceptable for a function which returns size_t, so we
// define our own.
//
// n is the dividend and m is the divisor
//
// Returns a positive integer r = n (mod m), in the range
// 0 <= r < m.
static size_t modulo(const ssize_t n, const size_t m);

// Produces a mask which, when ANDed with a byte, retains only the
// bit_index th byte.
//
// Example: bitmask(5) produces the byte 0b00100000.
//
// (Note that here the index is counted from right
// to left, which is different from how we represent bitarrays in the
// tests.  This function is only used by bitarray_get and bitarray_set,
// however, so as long as you always use bitarray_get and bitarray_set
// to access bits in your bitarray, this reverse representation should
// not matter.
static char bitmask(const size_t bit_index);


// ******************************* Functions ********************************

bitarray_t *bitarray_new(const size_t bit_sz) {
    // Allocate an underlying buffer of ceil(bit_sz/8) bytes.
    const size_t bytes = (bit_sz + 7) / 8;
    char *buf = _mm_malloc(bytes, sizeof(pocket));
    if (mlock(buf, bytes) != 0) {
        perror("mlock");
        return NULL;
    }

    if (buf == NULL) {
        return NULL;
    }

    // Allocate space for the struct.
    bitarray_t *const bitarray = malloc(sizeof(struct bitarray));
    if (bitarray == NULL) {
        free(buf);
        return NULL;
    }

    bitarray->buf = buf;
    bitarray->bit_sz = bit_sz;
    return bitarray;
}

void bitarray_free(bitarray_t *const bitarray) {
    if (bitarray == NULL) {
        return;
    }
    munlock(bitarray->buf, (bitarray->bit_sz + 7) / 8);
    free(bitarray->buf);
    bitarray->buf = NULL;
    _mm_free(bitarray);
}

size_t bitarray_get_bit_sz(const bitarray_t *const bitarray) {
    return bitarray->bit_sz;
}

bool bitarray_get(const bitarray_t *const bitarray, const size_t bit_index) {
    assert(bit_index < bitarray->bit_sz);

    // We're storing bits in packed form, 8 per byte.  So to get the nth
    // bit, we want to look at the (n mod 8)th bit of the (floor(n/8)th)
    // byte.
    //
    // In C, integer division is floored explicitly, so we can just do it to
    // get the byte; we then bitwise-and the byte with an appropriate mask
    // to produce either a zero byte (if the bit was 0) or a nonzero byte
    // (if it wasn't).  Finally, we convert that to a boolean.
    return (bitarray->buf[bit_index / 8] & bitmask(bit_index)) ? true : false;
}

void bitarray_set(bitarray_t *const bitarray,
                  const size_t bit_index,
                  const bool value) {
    assert(bit_index < bitarray->bit_sz);

    // We're storing bits in packed form, 8 per byte.  So to set the nth
    // bit, we want to set the (n mod 8)th bit of the (floor(n/8)th) byte.
    //
    // In C, integer division is floored explicitly, so we can just do it to
    // get the byte; we then bitwise-and the byte with an appropriate mask
    // to clear out the bit we're about to set.  We bitwise-or the result
    // with a byte that has either a 1 or a 0 in the correct place.
    bitarray->buf[bit_index / 8] =
            (bitarray->buf[bit_index / 8] & ~bitmask(bit_index)) |
            (value ? bitmask(bit_index) : 0);
}

void bitarray_randfill(bitarray_t *const bitarray) {
    for (int64_t i = 0; i < (bitarray->bit_sz + 7) / 8; i++) {
        bitarray->buf[i] = (char) (rand() & 0xFF);
    }
}

void bitarray_rotate(bitarray_t *const bitarray,
                     const size_t bit_offset,
                     const size_t bit_length,
                     const ssize_t bit_right_amount) {
    assert(bit_offset + bit_length <= bitarray->bit_sz);

    if (bit_length == 0) {
        return;
    }

    // Convert a rotate left or right to a left rotate only, and eliminate
    // multiple full rotations.
    bitarray_rotate_left(bitarray, bit_offset, bit_length,
                         modulo(-bit_right_amount, bit_length));
}

alignas(sizeof(pocket)) static constexpr uint8_t rev_bytes_512_raw[sizeof(pocket)]




=
 {
    63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48,
    47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32,
    31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16,
    15, 14, 13, 12, 11, 10,  9,  8,  7,  6,  5,  4,  3,  2,  1,  0
};

static pocket reverse_pocket(const pocket x) {
#if defined(__GFNI__) && defined(__SSSE3__) && defined(__AVX512VL__) && defined(__AVX512VBMI__) && defined(__BMI2__)
    const pocket bits_level_reverse_mask = _mm512_set1_epi64(0x8040201008040201ll);
    const pocket bits_level_reversed = _mm512_gf2p8affine_epi64_epi8(x, bits_level_reverse_mask, 0);
    const pocket byte_level_reverse_mask = _mm512_loadu_si512(rev_bytes_512_raw);
    return _mm512_permutexvar_epi8(byte_level_reverse_mask, bits_level_reversed);
#else
    static_assert(0, "cpu must support avx-512");
#endif
}

static pocket left_rotate(const pocket x, const size_t n) {
    assert(n > 0);
    assert(n < pocket_bit_size);
    const size_t q = n / word_bit_size, r = n % word_bit_size;

    const pocket byte_level_rotate_mask = _mm512_setr_epi64(-q & 7, 1 - q & 7, 2 - q & 7, 3 - q & 7, 4 - q & 7,
                                                            5 - q & 7, 6 - q & 7, 7 - q & 7);
    const pocket byte_level_rotate_shifted_mask = _mm512_setr_epi64(7 - q & 7, -q & 7, 1 - q & 7, 2 - q & 7,
                                                                    3 - q & 7, 4 - q & 7, 5 - q & 7, 6 - q & 7);

    const pocket byte_level_rotated = _mm512_permutexvar_epi64(byte_level_rotate_mask, x);
    const pocket byte_level_rotated_shifted = _mm512_permutexvar_epi64(byte_level_rotate_shifted_mask, x);
    return _mm512_shldv_epi64(byte_level_rotated, byte_level_rotated_shifted, _mm512_set1_epi64(r));
}

static pocket make_ones(const size_t n) {
    const size_t q = n / word_bit_size, r = n % word_bit_size;
    const pocket full = _mm512_maskz_set1_epi64(_bzhi_u64(~0ll, q), -1ll);
    return _mm512_mask_set1_epi64(full, 1ull << q, _bzhi_u64(~0ll, r));
}

static size_t pocket_reverse(pocket *arr, const size_t shift, const size_t cnt) {
    size_t iters = 0;
    if (shift == 0) {
        for (; 2 * iters + 1 < cnt; iters++) {
            const pocket right = reverse_pocket(arr[cnt - iters - 1]);
            arr[cnt - iters - 1] = reverse_pocket(arr[iters]);
            arr[iters] = right;
        }
        return iters;
    }

    if (cnt < 3) {
        return 0;
    }

    const pocket shift_ones_mask = make_ones(shift);
    pocket right_old = arr[cnt], reright_old = left_rotate(reverse_pocket(right_old), shift);
    __asm__ volatile("# LLVM-MCA-BEGIN hot_loop");
#pragma clang loop unroll_count(4)
    for (; iters < (cnt - 1) / 2; iters++) {
        const pocket left = left_rotate(reverse_pocket(arr[iters]), shift);
        const pocket right = arr[cnt - iters - 1];
        const pocket reright = left_rotate(reverse_pocket(right), shift);

        arr[cnt - iters] = _mm512_ternarylogic_epi64(shift_ones_mask, left, right_old, 0xCA); // 0xCA = !A & C | A & B
        arr[iters] = _mm512_ternarylogic_epi64(shift_ones_mask, reright, reright_old, 0xAC);
        right_old = _mm512_ternarylogic_epi64(shift_ones_mask, left, right, 0xAC); // 0xAC = A & C | !A & B
        reright_old = reright;
    }
    __asm__ volatile("# LLVM-MCA-END hot_loop");

    arr[cnt - iters] = right_old;
    return iters;
}

static void bitarray_reverse(bitarray_t *const bitarray, const size_t bit_offset, const size_t bit_length) {
    if (bit_length <= 1) {
        return;
    }

    size_t left_bound = bit_offset, right_bound = bit_offset + bit_length - 1;
    for (; left_bound < right_bound && left_bound % pocket_bit_size; left_bound++, right_bound--) {
        const bool bit = bitarray_get(bitarray, left_bound);
        bitarray_set(bitarray, left_bound, bitarray_get(bitarray, right_bound));
        bitarray_set(bitarray, right_bound, bit);
    }

    if (left_bound >= right_bound) {
        return;
    }

    pocket *arr = (pocket *) bitarray->buf + left_bound / pocket_bit_size;
    const size_t shift = (right_bound - left_bound + 1) % pocket_bit_size;
    const size_t cnt = (right_bound - left_bound + 1) / pocket_bit_size;

    const size_t iters = pocket_reverse(arr, shift, cnt);
    left_bound += iters * pocket_bit_size;
    right_bound -= iters * pocket_bit_size;

    for (; left_bound < right_bound; left_bound++, right_bound--) {
        const bool bit = bitarray_get(bitarray, left_bound);
        bitarray_set(bitarray, left_bound, bitarray_get(bitarray, right_bound));
        bitarray_set(bitarray, right_bound, bit);
    }
}

static void bitarray_rotate_left(bitarray_t *const bitarray,
                                 const size_t bit_offset,
                                 const size_t bit_length,
                                 const size_t bit_left_amount) {
    bitarray_reverse(bitarray, bit_offset, bit_left_amount);
    bitarray_reverse(bitarray, bit_offset + bit_left_amount, bit_length - bit_left_amount);
    bitarray_reverse(bitarray, bit_offset, bit_length);
}

static size_t modulo(const ssize_t n, const size_t m) {
    const ssize_t signed_m = (ssize_t) m;
    assert(signed_m > 0);
    const ssize_t result = ((n % signed_m) + signed_m) % signed_m;
    assert(result >= 0);
    return (size_t) result;
}

static char bitmask(const size_t bit_index) {
    return 1 << (bit_index % 8);
}
