#!/bin/bash
# author: otavio 
#
# Please, first read ./README
#
# Script to generate values_importance.ini

echo "%include values_wa.ini" > values_importance.ini
echo "" >> values_importance.ini
echo "[data_vector]" >> values_importance.ini
for ((i=1; i<=$1; i++))
do
		echo "2pt_theory_$i = 0 0.5 1" >> values_importance.ini
done
