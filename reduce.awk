#!/usr/bin/awk -f
#
# author: Otavio Alves
# This script removes data_vector columns from a chain file. This is done mainly to reduce the filesize.
#

# Looking at the header of the file, creates an array with the indices we want to print
NR == 1 {
	# Workaround to declare empty array
	split("", indices);
	
	# Look for columns that are not data_vector and append to the indices array
	for(i = 1; i <= NF; i++) {
		if($i !~ /^data_vector/){
			indices[length(indices) + 1] = i;
			printf "%s\t", $i;
		}
	}
	printf "\b\n";
}

# If line is a comment, print it as it is
/^#/ && NR > 1 {print}

/^[^#]/ {
	# Print only columns in the indices array
	for(i in indices){
		printf "%s\t", $i;
	}
	printf "\b\n";
}
