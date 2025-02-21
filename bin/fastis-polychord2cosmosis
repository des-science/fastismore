#!/usr/bin/env python

import numpy as np
import sys, os, argparse
from tqdm import tqdm

def parse_polychord_to_cosmosis(cosmosis_filename, polychord_filename, output_filename):
    if os.path.exists(output_filename):
        print(f"WARNING: output file {output_filename} already exists. Skipping.")
        return

    if not os.path.exists(cosmosis_filename):
        print(f"WARNING: cosmosis file {cosmosis_filename} does not exist. Skipping.")
        return
    
    if not os.path.exists(polychord_filename):
        print(f"WARNING: polychord file {polychord_filename} does not exist. Skipping.")
        return

    with open(output_filename, 'w') as output:
        # Loads labels from cosmosis' file
        with open(cosmosis_filename, 'r') as f:
            labels_cosmosis = np.array(f.readline()[1:-1].lower().split())
            mask_cosmosis = [l not in ['weight', 'like', 'prior', 'post'] for l in labels_cosmosis]

            # Order of columns in polychord output files
            labels_pc = np.array(['weight', 'like'] + list(labels_cosmosis[mask_cosmosis]) + ['prior']) 
            mask_pc = [("data_vector" not in l) and ("sigma_crit_inv_lens_source" not in l) for l in labels_pc]

            # Write header with labels
            output.write('#' + '\t'.join(labels_pc[mask_pc]) + '\tpost\n')

            print("Copying comments from cosmosis file...")
            for line in f.readlines():
                if '#' in line:
                    output.write(line)
            print("Done!")

        with open(polychord_filename, 'r') as f:
            print("Copying samples from polychord file...")
            for line in tqdm(f.readlines()):

                # Do not output polychord comments (if any)
                if '#' in line:
                    continue 

                data = np.array(line.split())[mask_pc].tolist()
                like = -0.5*np.double(data[1])
                data[1] = str(like)
                prior = np.double(data[-1])
                data.append(str(prior + like)) # post
                output.write('\t'.join(data) + '\n')

            print("Done!")

def parse_polychord_to_cosmosis_with_is(cosmosis_filename, polychord_filename, is_filename, output_filename, override=False):
    if os.path.exists(output_filename):
        print(f"WARNING: output file {output_filename} already exists. {'Overriding' if override else 'Skipping'}.")
        if not override:
            return

    if not os.path.exists(cosmosis_filename):
        print(f"WARNING: cosmosis file {cosmosis_filename} does not exist. Skipping.")
        return
    
    if not os.path.exists(polychord_filename):
        print(f"WARNING: polychord file {polychord_filename} does not exist. Skipping.")
        return
    
    if not os.path.exists(is_filename):
        print(f"WARNING: IS file {is_filename} does not exist. Skipping.")
        return
    
    with open(output_filename, 'w') as output:
        # Loads labels from cosmosis' file
        with open(cosmosis_filename, 'r') as f:
            labels_cosmosis = np.array(f.readline()[1:-1].lower().split())
            mask_cosmosis = [l not in ['weight', 'like', 'prior', 'post'] for l in labels_cosmosis]

            # Order of columns in polychord output files
            labels_pc = np.array(['old_weight', 'old_like'] + list(labels_cosmosis[mask_cosmosis]) + ['prior']) 
            mask_pc = [("data_vector" not in l) and ("sigma_crit_inv_lens_source" not in l) for l in labels_pc]

            # Write header with labels
            output.write('#' + '\t'.join(labels_pc[mask_pc]) + '\tlike\told_post\tpost\tlog_weight\tweight\n')

            print("Copying comments from cosmosis file...")
            for line in f.readlines():
                if '#' in line:
                    output.write(line)
            print("Done!")

        # Will copy new_like and weights from importance sampling weights file
        try:
            old_like_is, old_weight_is, new_like_is, weight_is = np.loadtxt(is_filename, unpack=True)
        except:
            print("ERROR: is the file empty?")
            return

        new_like_iter = iter(new_like_is)
        old_like_iter = iter(old_like_is)
        weight_iter = iter(weight_is)

        with open(polychord_filename, 'r') as f:
            print("Copying samples from polychord file...")
            for line in tqdm(f.readlines()):

                # Do not output polychord comments (if any)
                if '#' in line:
                    continue 

                data = np.array(line.split())[mask_pc].tolist()
                old_like = -0.5*np.double(data[1])
                data[1] = str(old_like)
                prior = np.double(data[-1])
                new_like_e = next(new_like_iter)
                old_like_e = next(old_like_iter)
                data.append(str(old_like - old_like_e + new_like_e)) # like
                data.append(str(prior + old_like)) # old_post
                data.append(str(prior + old_like - old_like_e + new_like_e)) # post
                data.append(str(new_like_e - old_like_e)) # log_weight
                data.append(str(next(weight_iter))) # weight
                output.write('\t'.join(data) + '\n')

            print("Done!")

        print("Copying comments from importance sampling weights file...")
        with open(is_filename, 'r') as f:
            f.readline() # throw away first line with labels
            for line in f.readlines():
                if '#' in line:
                    output.write(line)
        print("Done!")

def parse_cosmosis_with_is(cosmosis_filename, is_filename, output_filename):
    if os.path.exists(output_filename):
        print(f"WARNING: output file {output_filename} already exists. Skipping.")
        return
    
    if not os.path.exists(is_filename):
        print(f"WARNING: IS file {is_filename} does not exist. Skipping.")
        return
    
    if not os.path.exists(cosmosis_filename):
        print(f"WARNING: cosmosis file {cosmosis_filename} does not exist. Skipping.")
        return
    
    with open(output_filename, 'w') as output:
        # Loads labels from cosmosis' file
        with open(cosmosis_filename, 'r') as f:
            labels_cosmosis = np.array(f.readline()[1:-1].lower().split())
            mask_cosmosis = [(l not in ['weight', 'like', 'post', 'old_weight', 'old_post', 'log_weight']) and ("data_vector" not in l) and ("sigma_crit_inv_lens_source" not in l) for l in labels_cosmosis]
            cosmo_dict = {l: i for i,l in enumerate(labels_cosmosis)}


            # Order of columns in polychord output files
            mask_dv = [("data_vector" not in l) and ("sigma_crit_inv_lens_source" not in l) for l in labels_cosmosis]

            # Write header with labels
            output.write('#' + '\t'.join(labels_cosmosis[mask_cosmosis]) + '\told_like\tlike\told_post\tpost\tlog_weight\tweight\n')

            print("Copying comments from cosmosis file...")
            for line in f.readlines():
                if '#' in line:
                    output.write(line)
            print("Done!")

            # Will copy new_like and weights from importance sampling weights file
            old_like_is, old_weight_is, new_like_is, weight_is = np.loadtxt(is_filename, unpack=True) 

            new_like_iter = iter(new_like_is)
            old_like_iter = iter(old_like_is)
            weight_iter   = iter(weight_is)
            
            f.seek(0)
    
            print("Copying samples from cosmosis file...")
            for line in tqdm(f.readlines()):

                # Do not output polychord comments (if any)
                if '#' in line:
                    continue 

                data = np.array(line.split())[mask_cosmosis].tolist()
                input_data = np.array(line.split()).tolist()

                original_like = np.double(input_data[cosmo_dict['like']])
                prior = np.double(input_data[cosmo_dict['prior']])
                new_like_e = next(new_like_iter)
                old_like_e = next(old_like_iter)
                data.append(input_data[cosmo_dict['like']]) # old_like
                data.append(str(original_like - old_like_e + new_like_e)) # like
                data.append(input_data[cosmo_dict['post']]) # old_post
                data.append(str(np.double(input_data[cosmo_dict['post']]) - old_like_e + new_like_e)) # post
                data.append(str(new_like_e - old_like_e)) # log_weight
                data.append(str(next(weight_iter))) # weight
                output.write('\t'.join(data) + '\n')

            print("Done!")

        print("Copying comments from importance sampling weights file...")
        with open(is_filename, 'r') as f:
            f.readline() # throw away first line with labels
            for line in f.readlines():
                if '#' in line:
                    output.write(line)
        print("Done!")

def main():
    parser = argparse.ArgumentParser(description = '')

    parser.add_argument('cosmosis_filename', help = 'base chain file name.')
    parser.add_argument('--polychord', dest = 'polychord_filename',
                        default = None, required = False,
                        help = 'polychord extra samples file name')
    parser.add_argument('--is-weights', dest = 'is_filename',
                        default = None, required = False,
                        help = 'importance weights filename')
    parser.add_argument('output_filename', help = 'output file name.')

    args = parser.parse_args()

    if   args.is_filename is None and args.polychord_filename is not None:
        parse_polychord_to_cosmosis(args.cosmosis_filename, args.polychord_filename, args.output_filename)
    elif args.is_filename is not None and args.polychord_filename is None:
        parse_cosmosis_with_is(args.cosmosis_filename, args.is_filename, args.output_filename)
    elif args.is_filename is not None and args.polychord_filename is not None:
        parse_polychord_to_cosmosis_with_is(args.cosmosis_filename, args.polychord_filename, args.is_filename, args.output_filename)
    else:
        print("What do you want to do? Specify polychord or IS weights.")

if __name__ == '__main__':
    main()
