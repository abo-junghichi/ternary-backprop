#!/bin/bash

#./a.out init

touch $0.lock
while [ -e $0.lock ]
do
	date
	printf "" > ma.log
	for i in `seq 0 59999 | shuf`
	do
		echo $i >> ma.log
		./a.out learn $i >> ma.log
		echo $? >> ma.log
	done
	cp brain.img brain.img.back
	grep '\]0$' < ma.log | wc -l
done
date
