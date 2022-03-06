#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include "mmapfile.c"
#include "neuron.c"
/* assume WORD_WIDTH=32 */
typedef struct {
    sfmt_t seed;
    synapse s0[320][17], s1[544][27], s2[864][45], s3[1440][73],
	s4[2336][120], s5[3840][196];
    inertia n0[320][17], n1[544][27], n2[864][45], n3[1440][73],
	n4[2336][120], n5[3840][196];
    unsigned char r0[320], r1[544], r2[864], r3[1440], r4[2336], r5[3840];
} brain;
int main(int argc, char **argv)
{
    int rtn = 0;
    int i;
    char mode;
    int target_id;
    mmapfile netfile;
    char brain_path[] = "brain.img";
    brain *net;
    word ir[10], i0[17], i1[27], i2[45], i3[73], i4[120], i5[196];
    int g0[320], g1[544], g2[864], g3[1440], g4[2336], g5[3840];
    int b0[320], b1[544], b2[864], b3[1440], b4[2336], b5[3840];
    int label;
    if (2 <= argc)
	mode = argv[1][0];
    else
	goto print_help;
    if ('i' != mode && 3 != argc) {
      print_help:
	fprintf(stderr, "usage: %s [init|learn|test] [target_id]\n",
		argv[0]);
	abort();
    }
    if ('t' == mode || 'l' == mode) {
	mmapfile labels, images;
	if ('l' == mode) {
	    labels =
		mmapfile_open("mnt/train-labels-idx1-ubyte", 1, 0, 60008);
	    images =
		mmapfile_open("mnt/train-images-idx3-ubyte", 1, 0,
			      47040016);
	} else {
	    labels =
		mmapfile_open("mnt/t10k-labels-idx1-ubyte", 1, 0, 10008);
	    images =
		mmapfile_open("mnt/t10k-images-idx3-ubyte", 1, 0, 7840016);
	}
	target_id = atoi(argv[2]);
	label = ((char *) labels.addr)[8 + target_id];
	assert(label < 10);
	for (i = 0; i < 28 * 28 / 4; i++)
	    i5[i] =
		((uint32_t *) images.addr)[4 + (target_id * 28 * 28 / 4) +
					   i];
	mmapfile_close(labels);
	mmapfile_close(images);
    } else if ('i' == mode) {
	uint32_t dummy = 0;
	FILE *netf = fopen(brain_path, "w");
	for (i = 0; i < sizeof(brain) / sizeof(uint32_t); i++)
	    fwrite(&dummy, sizeof(uint32_t), 1, netf);
	fclose(netf);
    } else
	abort();
    netfile = mmapfile_open(brain_path, 1, 1, sizeof(brain));
    net = netfile.addr;
    if ('i' == mode) {
	sfmt_init_gen_rand(&net->seed, 123456);
	sfmt_fill_array32(&net->seed, net->s0,
			  (offsetof(brain, n0) -
			   offsetof(brain, s0)) / sizeof(uint32_t));
    } else {
	unsigned int miss = 0, rst[10];
#define GUESS_CORE(layer,wc,nc) \
    for (i = 0; i < nc; i++)\
	g##layer[i] =\
	    propagation_guess_core(wc, net->s##layer[i], i##layer)
#define GUESS(cur,wc,nc,up) \
    GUESS_CORE(cur, wc, nc);\
    propagation_word_core(nc / WORD_WIDTH, g##cur, i##up)
	GUESS(5, 196, 3840, 4);
	GUESS(4, 120, 2336, 3);
	GUESS(3, 73, 1440, 2);
	GUESS(2, 45, 864, 1);
	GUESS(1, 27, 544, 0);
	GUESS(0, 17, 320, r);
	for (i = 0; i < 10; i++) {
	    rst[i] = popcount(ir[i]);
	    if (miss < rst[i] && i != label)
		miss = rst[i];
	    printf("%i ", rst[i]);
	}
	for (i = 0; i < 10; i++) {
	    int n, back = 0;
	    if (i == label)
		back = 1;
	    else if (rst[i] >= miss)
		back = -1;
	    for (n = 0; n < WORD_WIDTH; n++)
		b0[i * WORD_WIDTH + n] = back;
	}
	printf(":%u-%u %i\n", miss, rst[label], label);
	if (miss >= rst[label])
	    rtn = 1;
	if ('l' == mode) {
#define BACK(cur,under,wc,nc) \
	    {\
		unsigned int edge = 0, flat = 0, acc = 0;\
		unsigned int dw = wc * WORD_WIDTH;\
		dw = delta_width_core(dw, nc, g##cur, b##cur);\
		if (NULL != under)\
		    for (i = 0; i < wc * WORD_WIDTH; i++)\
			((int *) under)[i] = 0;\
		for (i = 0; i < nc; i++)\
		    switch (backpropagation_core\
			    (0, &net->seed, dw, net->r##cur + i, g##cur[i],\
			     b##cur[i], wc, i##cur, net->s##cur[i],\
			     net->n##cur[i], under)) {\
		    case 1:\
			edge++;\
			break;\
		    case 2:\
			acc++;\
			break;\
		    case 3:\
			flat++;\
			break;\
		    }\
		printf("[%u,%u,%u,%u]", dw, edge, acc, flat);\
	    }
	    BACK(0, b1, 17, 320);
	    BACK(1, b2, 27, 544);
	    BACK(2, b3, 45, 864);
	    BACK(3, b4, 73, 1440);
	    BACK(4, b5, 120, 2336);
	    BACK(5, NULL, 196, 3840);
	}
    }
    mmapfile_close(netfile);
    return rtn;
}
