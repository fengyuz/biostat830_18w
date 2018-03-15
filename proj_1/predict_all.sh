#!/bin/bash
export WORKON_HOME=$HOME/.virtualenvs
source /usr/local/bin/virtualenvwrapper.sh
workon tensorflow
allPred=predicted_pheno/ALL_predicted_pheno.dat
rm -f ${allPred}
for train_path in training_geno/*training_geno.dat; do
	date
	echo Predicting from ${train_fname} ...
	train_fname=${train_path#*/} 
	gene=${train_fname%.*}
	gene=${gene%.*}
	pred_path=predicted_pheno/${gene}.predicted_pheno.dat
	python predict.py ${train_fname}
	echo -n ${gene} >> ${allPred}
	echo -ne '\t' >> ${allPred}
	cat ${pred_path} | awk 1 ORS='\t' >> ${allPred}
	echo >> ${allPred}
	echo Done!
done
