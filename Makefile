CC=gcc -O3 -Wall -Wextra -pedantic -Wstrict-aliasing=1
#CC=gcc -O0 -Wall -Wextra -pedantic -Wstrict-aliasing=1
TMP_BRAIN=/tmp/brain-tmp.img

all: test
mrproper-all: mrproper
	rm mnt/t10k-images-idx3-ubyte mnt/train-images-idx3-ubyte mnt/t10k-labels-idx1-ubyte mnt/train-labels-idx1-ubyte
mrproper: clean
	rm learn.c brain.img brain.img.back ma.log
clean:
	rm a.out $(TMP_BRAIN)

mnt/t10k-images-idx3-ubyte:
	wget 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz' -O - | gunzip > mnt/t10k-images-idx3-ubyte
mnt/train-images-idx3-ubyte:
	wget 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz' -O - | gunzip > mnt/train-images-idx3-ubyte
mnt/t10k-labels-idx1-ubyte:
	wget 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz' -O - | gunzip > mnt/t10k-labels-idx1-ubyte
mnt/train-labels-idx1-ubyte:
	wget 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz' -O - | gunzip > mnt/train-labels-idx1-ubyte

learn.c:
	ln -s learn-l3.c learn.c
a.out: learn.c
	$(CC) learn.c -o a.out
brain.img:
	ln -s $(TMP_BRAIN) brain.img
$(TMP_BRAIN): a.out brain.img
	./a.out i
test: $(TMP_BRAIN) mnt/t10k-images-idx3-ubyte mnt/train-images-idx3-ubyte mnt/t10k-labels-idx1-ubyte mnt/train-labels-idx1-ubyte
	bash test.bash
