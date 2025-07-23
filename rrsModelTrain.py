import numpy as np

def rrsModelTrain(daph, pft, pft_index, n_permutations, max_pcs, k, mdl_pick_metric, output_file_name):
    coefficients = None
    intercepts = None
    summary_gofs = None
    all_gofs = None

    # check for NaNs
    if np.isnan(pft).any():
        print('pft data has NaNs. remove them and try again plz')
        return coefficients, intercepts, summary_gofs, all_gofs
    elif np.isnan(daph).any():
        print('spectral data has NaNs. remove them and try again plz')
        return coefficients, intercepts, summary_gofs, all_gofs
    
    # check that daph and pft have the same number of rows
    if daph.shape[0] != pft.shape[0]:
        print('daph and pft must have the same number of rows')
        return coefficients, intercepts, summary_gofs, all_gofs
    
    max_components = max_pcs
    spectra_4_mdl = daph

    # set random number generator seed for reproducibility
    np.random.seed(42)

    for i in range(n_permutations):
        # Create broad training data (75%) and validation data (25%)
        training_indices = np.random.permutation(len(pft))[:int(len(pft) * 0.75)]
        pigs_training = pft[training_indices]
        spectra_4_mdl_training = spectra_4_mdl[training_indices,:]

        # validation data
        pigs_validate = pft
        pigs_validate = np.delete(pigs_validate,training_indices)
        spectra_4_mdl_validate = spectra_4_mdl
        spectra_4_mdl_validate = np.delete(spectra_4_mdl_validate, training_indices, axis=0)

        # get set up for k-fold cross validation

        pig_len = len(pigs_training)
        train_split = pig_len/k

        rand_ns = np.random.permutation(pig_len)
        CV_indices = np.full((k, int(np.ceil(train_split))+1), np.nan)
        n_leftovers = int(np.round((train_split - (int(train_split))) * k))
        counter_start = n_leftovers + 1
        counter_end = n_leftovers + int(train_split)

        for j in range(k):
            CV_indices[j, 0:int(train_split)] = rand_ns[counter_start-1:counter_end]
            counter_start += int(train_split)
            counter_end += int(train_split)

        # add the leftovers to the CV_indices array, and put NaN's for the sets
        # where there's not enough leftovers to go around
        leftovers = rand_ns[:n_leftovers-1]
        # Pad with NaNs if leftovers < k
        if len(leftovers) < k:
            pad = np.full(k - len(leftovers), np.nan)
            leftovers = np.concatenate([leftovers, pad])
        # Add a new column (grow to k x (num_folds + 1))
        leftovers = leftovers.reshape(k, 1)
        CV_indices = np.hstack([CV_indices, leftovers])
        # so now CV indices contains k sets of random indices for k-fold CV


        for j in range(k):
            # Split up CV data sets
            these_CV_indices = CV_indices[j, :]
            these_CV_indices = these_CV_indices[~np.isnan(these_CV_indices)].astype(int)
            CV_valid_pigs = pigs_training[these_CV_indices]
            CV_valid_spec = spectra_4_mdl_training[these_CV_indices, :]
            CV_train_pigs = np.delete(pigs_training, these_CV_indices, axis=0)
            CV_train_spec = np.delete(spectra_4_mdl_training, these_CV_indices, axis=0)
            
            '''
            shapes are aligned up to this point
            '''


    return coefficients, intercepts, summary_gofs, all_gofs