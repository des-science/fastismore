#!/bin/bash
# author: otavio 
#
# Please, first read ./README
#
# Script to rewrite data_vector--2pt_theory in the header of a chain
# Assumes all columns without name are related to data_vector--2pt_theory
# 
# data_vector--2pt_theory -> data_vector--2pt_theory_i,
# i = {1..len(n_columns - n_columns_header + 1)}
#
# Also generates values_importance.ini

cp $1 $1.new
n_head=$(head -n1 $1.new | tr '\t' '\n' | wc -l)
n_tail=$(tail -n1 $1.new | tr '\t' '\n' | wc -l)
((diff = $n_tail - $n_head + 1))
string=$(awk -v d="$diff" 'BEGIN{for(c=1;c<=d;c++) printf "DATA_VECTOR--2PT_THEORY_"c"	";}')
sed -i "1s/DATA_VECTOR--2PT_THEORY\t/$string/" $1.new

echo "%include values_\${RUN_NAME_STR}.ini" > values_importance.ini
echo "" >> values_importance.ini
echo "[data_vector]" >> values_importance.ini
for ((i=1; i<=$diff; i++))
do
		echo "2pt_theory_$i = -10 0 10" >> values_importance.ini
done
