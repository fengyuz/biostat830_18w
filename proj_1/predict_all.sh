#!/bin/bash
export WORKON_HOME=$HOME/.virtualenvs
source /usr/local/bin/virtualenvwrapper.sh
workon tensorflow
for filename in training_geno/*training_geno.dat; do
	fname=${filename#*/} 
	date
	echo Predicting from ${fname} ...
	python predict.py ${fname}
	echo Done!
done
