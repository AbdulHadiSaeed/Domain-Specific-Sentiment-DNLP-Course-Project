#!/bin/bash
for dir in ./*
do
	if [ -d "$dir" ]
	then
		echo $dir
		cd $dir
		cat train.tsv | tr '[:space:]' '[\n*]' | grep -v "^\s*$" | grep -v '^[[:punct:]]*$' | sort | uniq -c | sort -bnr > train.vocab
		cat test.tsv | tr '[:space:]' '[\n*]' | grep -v "^\s*$" | grep -v '^[[:punct:]]*$' |sort | uniq -c | sort -bnr > test.vocab
		cat dev.tsv | tr '[:space:]' '[\n*]' | grep -v "^\s*$" | grep -v '^[[:punct:]]*$' |sort | uniq -c | sort -bnr > dev.vocab
		cat train.tsv test.tsv dev.tsv| tr '[:space:]' '[\n*]' | grep -v "^\s*$" | grep -v '^[[:punct:]]*$' |sort | uniq -c | sort -bnr > all.vocab
		cd ..
	fi
done
