from __future__ import print_function
from builtins import zip
from builtins import map
from builtins import str
from builtins import range
import itertools
import numpy as np
from cosmosis.output.text_output import TextColumnOutput
from .. import ParallelSampler


def task(p):
    i,p = p
    results = list_sampler.pipeline.run_results(p, all_params=True)
    #If requested, save the data to file
    if list_sampler.save_name and results.block is not None:
        results.block.save_to_file(list_sampler.save_name+"_%d"%i, clobber=True)
    return results.post, (results.prior, results.extra)




class ListSampler(ParallelSampler):
    parallel_output = False
    sampler_outputs = [("prior", float), ("post", float)]

    def config(self):
        global list_sampler
        list_sampler = self

        self.converged = False
        self.filename = self.read_ini("filename", str)
        self.save_name = self.read_ini("save", str, "")
        self.burn = self.read_ini("burn", int, 0)
        self.thin = self.read_ini("thin", int, 1)
        limits = self.read_ini("limits", bool, False)

        #overwrite the parameter limits
        if not limits:
            if self.output is not None:
                self.output.columns = []
            for p in self.pipeline.parameters:
                p.limits = (-np.inf, np.inf)
                if self.output is not None:
                    self.output.add_column(str(p), float)
            if self.output is not None:
                for p in self.pipeline.extra_saves:
                # SJ begin ----------------------------------------------
                # *** from otavio 's code   
                # modification for importance sampling
                # This transforms the name data_vector/2pt_theory#457 into
                    # data_vector/2pt_theory_1, data_vector/2pt_theory_2, ... 
                #
                # original code : 
                # self.output.add_column('{}--{}'.format(*p), float)
                #
                # modification : 
                    if ('#' in p[1]):
                        n,l = p[1].split('#')
                        for i in range(1,int(l)+1):
                                #extra_names.append('%s--%s_%d'%(section,n,i))
                            self.output.add_column('%s--%s_%d'%(p[0],n,i), float)
                    else:
                        #pass
                        #extra_names.append('%s--%s'%(section,name))
                            self.output.add_column('{}--{}'.format(*p), float)
                self.output.add_column('weight', float) ## NW
                # SJ end ------------------------------------------------
                for p,ptype in self.sampler_outputs:
                    self.output.add_column(p, ptype)


    def execute(self):

        #Load in the filename that was requested
        file_options = {"filename":self.filename}
        column_names, samples, _, _, _ = TextColumnOutput.load_from_options(file_options)
        samples = samples[0]
        ix_weight = -1 #assume no weight column ## NW
        post_old = None # as a check, e.g. when running a list_sampler on a baseline chain to get theory values, expect same post ##NW
        # find where in the parameter vector of the pipeline
        # each of the table parameters can be found
        replaced_params = []
        for i,column_name in enumerate(column_names):
            # ignore additional columns like e.g. "like", "weight"
            try:
                section,name = column_name.split('--')
            except ValueError:
                ## NW start --- if weight column present in list, propogate it to output
                if column_name == 'weight':
                    print('Found weight column in input list. Copying to output chain.')
                    ix_weight = i
                    weights = samples[:, ix_weight]
                elif column_name == 'post':
                    post_old = samples[:, i]
                ## NW end -------
                    print("Not including column %s as not a cosmosis name" % column_name)
                continue
            section = section.lower()
            name = name.lower()
            # find the parameter in the pipeline parameter vector
            # may not be in there - warn about this
            try:
                j = self.pipeline.parameters.index((section,name))
                replaced_params.append((i,j))
            except ValueError:
                print("Not including column %s as not in values file" % column_name)

        #Create a collection of sample vectors at the start position.
        #This has to be a list, not an array, as it can contain integer parameters,
        #unlike most samplers
        v0 = self.pipeline.start_vector(all_params=True, as_array=False)
        
        if ix_weight != -1: #if weight column, only run at points where weight is above minweight
            minweight = 1.e-9
            usebool = (samples[:,ix_weight] >= minweight)
            print('Only sampling at points with weight > {}'.format(minweight))
            print('{:.2g}% of samples dropped.'.format(usebool.sum()*100./len(samples)))
            samples = samples[usebool]
            weights = samples[:, ix_weight]
        else:
            usebool = np.ones(len(samples), dtype=bool)
        post_old = post_old[usebool]
        
        sample_vectors = [v0[:] for i in range(len(samples))]

        #Fill in the varied parameters. We are not using the
        #standard parameter vector in the pipeline with its 
        #split according to the ini file
        for s, v in zip(samples, sample_vectors):
            for i,j in replaced_params:
                v[j] = s[i]

        #Turn this into a list of jobs to be run 
        #by the function above
        sample_index = list(range(len(sample_vectors)))
        jobs = list(zip(sample_index, sample_vectors))

        #Run all the parameters
        #This only outputs them all at the end
        #which is a bit problematic, though you 
        #can't use MPI and retain the output ordering.
        #Have a few options depending on whether
        #you care about this we should think about
        #(also true for grid sampler).
        if self.pool:
            results = self.pool.map(task, jobs)
        else:
            results = list(map(task, jobs))
        #Save the results of the sampling
        #We now need to abuse the output code a little.
        ### NW start
        post_new = np.zeros(len(samples))
        if ix_weight == -1: # no input weight column so equally weight all samples in list
            weights = np.ones(len(samples))*1./len(samples)
        for i, (sample, result) in enumerate(zip(sample_vectors, results)):
            #Optionally save all the results calculated by each
            #pipeline run to files
            (prob, (prior,extra)) = result
            post_new[i] = prob
            #always save the usual text output
            self.output.parameters(sample, extra, weights[i], prior, prob)
        if post_old is not None: #alert if old and new posteriors don't match
            if not np.allclose(post_new, post_old):
                print('Warning: found posteriors in input list that dont match new posteriors!')
                print('Mean weighted difference of ln(post): ', np.average(post_new - post_old, weights=weights))
                print('RMS weighted difference of ln(post): ', np.average((post_new - post_old)**2, weights=weights)**0.5)
            else:
                'Passed: old and new posterior values agree.'
        ### NW end
        #We only ever run this once, though that could 
        #change if we decide to split up the runs
        self.converged = True

    def is_converged(self):
        return self.converged
