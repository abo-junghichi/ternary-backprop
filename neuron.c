#include <stddef.h>
#include <stdint.h>
#include <assert.h>
#include "sfmt/SFMT.c"
#define WORD_WIDTH 32
#define WORD_TYPE_core(width) uint##width##_t
#define WORD_TYPE(num_macro) WORD_TYPE_core(num_macro)
typedef WORD_TYPE(WORD_WIDTH) word;
static word popcount(word w)
{
/* https://en.wikipedia.org/wiki/Hamming_weight */
    word b1;
    switch (WORD_WIDTH) {
    case 8:
	w -= (w >> 1) & 0x55;
	b1 = 0x33;
	w = ((w >> 2) & b1) + (w & b1);
	w = ((w >> 4) + w) & 0x0f;
	break;
    case 32:
	w -= (w >> 1) & 0x55555555;
	b1 = 0x33333333;
	w = ((w >> 2) & b1) + (w & b1);
	w = ((w >> 4) + w) & 0x0f0f0f0f;
	w = (w >> 8) + w;
	w = ((w >> 16) + w) & 0x3f;
	break;
    case 64:
	w -= (w >> 1) & 0x5555555555555555;
	b1 = 0x3333333333333333;
	w = ((w >> 2) & b1) + (w & b1);
	w = ((w >> 4) + w) & 0x0f0f0f0f0f0f0f0f;
	w = (w >> 8) + w;
	w = (w >> 16) + w;
	w = ((w >> 32) + w) & 0x7f;
	break;
    default:
	assert(0);
    }
    return w;
}
typedef word synapse[2];
static int propagation_guess_core(size_t word_count, synapse * sy_array,
				  word * input_array)
{
    size_t i;
    int guess = 0;
    for (i = 0; i < word_count; i++) {
	word input = input_array[i];
	word rst = (input & sy_array[i][0]) ^ (input | sy_array[i][1]);
	guess += popcount(rst) - WORD_WIDTH / 2;
    }
    return guess;
}
/*
& |    0 1
0 0 -> 0 1
0 1 -> 1 1
1 0 -> 0 0
1 1 -> 1 0
*/
static void propagation_word_core(size_t word_count, int *guess_array,
				  word * output_array)
{
    size_t i;
    for (i = 0; i < word_count; i++) {
	int n;
	word rtn = 0;
	for (n = 0; n < WORD_WIDTH; n++)
	    rtn |= (0 <= *(guess_array++)) << n;
	output_array[i] = rtn;
    }
}
static word word_random(sfmt_t * random_state)
{
    switch (WORD_WIDTH) {
    case 8:
    case 32:
	return sfmt_genrand_uint32(random_state);
    case 64:
	return sfmt_genrand_uint64(random_state);
    default:
	assert(0);
	return 0;
    }
}
static unsigned int guess_abs(int guess)
{
    int mask = (0 <= guess) - 1;
    return guess ^ mask;
}
static unsigned int delta_width_core(unsigned int delta_width_max,
				     size_t neuron_count, int *guess_array,
				     int *back_array)
{
    size_t i;
    unsigned int rtn = delta_width_max;
    for (i = 0; i < neuron_count; i++)
	if (0 >= rtn)
	    break;
	else {
	    int guess = guess_array[i];
	    unsigned int cur_abs = guess_abs(guess);
	    if (cur_abs < rtn) {
		int back = back_array[i];
		if (0 != back && (0 <= guess) != (0 <= back))
		    rtn = cur_abs;
	    }
	}
    return rtn;
}
typedef unsigned char inertia[WORD_WIDTH][2];
static int backpropagation_core(int cold_level, sfmt_t * random_state,
				unsigned int delta_width,
				unsigned char *refresh, int guess,
				int back, size_t word_count,
				word * input_array, synapse * sy_array,
				inertia * ine_array, int *back_array)
{
    int rtn;
    size_t i;
    int n;
    int is_back;
    int ine_log2 = 8, cold = cold_level - ine_log2, hot_weight = 1;
    word sext = (0 <= back) - 1;
    if (0 == back || (0 <= guess) == (0 <= back))
	return 0;
    else if (guess_abs(guess) <= delta_width)
	rtn = 1;
    else if (255 > *refresh) {
	(*refresh)++;
	return 2;
    } else
	rtn = 3;
    *refresh = 0;
    {
	size_t min = guess_abs(guess) + 1, cur =
	    min + word_count * (WORD_WIDTH / 2) - (0 <= guess);
	for (; min < cur; cold++)
	    min <<= 1;
	if (1 > cold) {
	    /* hot_weight <<= 1 - cold; */
	    cold = 1;
	}
    }
    is_back = NULL != back_array && 1 == rtn;
    for (i = 0; i < word_count; i++) {
	int t;
	word heat, wand, wor;
	heat = word_random(random_state);
	for (t = 1; t < cold && 0 != heat; t++)
	    heat &= word_random(random_state);
	if (is_back || 0 != heat) {
	    wand = sy_array[i][0];
	    wor = sy_array[i][1];
	}
	if (is_back) {
	    word along = ~(wand | wor), against = wand & wor, plus =
		(along & ~sext) | (against & sext), minus =
		(along & sext) | (against & ~sext);
	    for (n = 0; n < WORD_WIDTH; n++)
		back_array[i * WORD_WIDTH + n] +=
		    (plus >> n & 0x1) - (minus >> n & 0x1);
	}
	if (0 != heat) {
	    word input = input_array[i], same, acc;
	    same = (input & wand) ^ (input | wor) ^ sext;
	    acc = same & heat;
	    for (n = 0; n < WORD_WIDTH; n++)
		if (acc >> n & 0x1) {
		    t = ~input >> n & 0x1;
		    if ((0x1 << ine_log2) - hot_weight >
			ine_array[i][n][t])
			ine_array[i][n][t] += hot_weight;
		}
	    acc = ~same & heat;
	    for (n = 0; n < WORD_WIDTH; n++)
		if (acc >> n & 0x1) {
		    t = ~input >> n & 0x1;
		    if (hot_weight - 1 < ine_array[i][n][t])
			ine_array[i][n][t] -= hot_weight;
		    else
			sy_array[i][t] ^= 0x1 << n;
		}
	}
    }
    return rtn;
}
