#!/bin/bash
# author: otavio 
#
# Please, first read ./README
#
# Script to generate values_importance.ini

echo "%include values_\${RUN_NAME_STR}.ini" > values_importance.ini
echo "" >> values_importance.ini
echo "[data_vector]" >> values_importance.ini
for ((i=1; i<=$1; i++))
do
		echo "2pt_theory_$i = -10 0 10" >> values_importance.ini
done
