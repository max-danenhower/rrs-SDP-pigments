import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
import pandas as pd


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

    # preallocate coefficients array
    mean_betas_nonstd = np.zeros((spectra_4_mdl.shape[1], n_permutations))
    mean_alphas_nonstd = np.zeros(n_permutations)

    # preallocate statistics/metrics arrays
    R2s_final = np.zeros(n_permutations)
    RMSEs_final = np.zeros(n_permutations)
    pct_bias = np.zeros(n_permutations)
    pct_errors = np.zeros((n_permutations,k))
    med_pct_error = np.zeros(n_permutations)
    avg_pct_error = np.zeros(n_permutations)
    CI_pct_error = np.zeros(n_permutations)
    std_pct_error = np.zeros(n_permutations)
    mae_final = np.zeros(n_permutations)

    for i in range(n_permutations):
        # Create broad training data (75%) and validation data (25%)
        training_indices = np.random.permutation(len(pft))[:int(len(pft) * 0.75)]

        # hard code for now
        #training_indices = np.array([18,10,5,1,6,9,7,2,14,4,13,17,8,3])-1  # Example indices for training, -1 for 0 indexing

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

        # hard code for now
        #rand_ns = np.array([11, 3, 1, 8, 4, 6, 9, 7, 14, 12, 10, 2, 5, 13]) - 1  # -1 for 0 indexing

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

        # preallocate arrays
        n_modes_to_use = np.zeros(k, dtype=int)
        betas = np.zeros((spectra_4_mdl_training.shape[1], k))
        alpha = np.zeros(k)
        CV_R2s = np.zeros(k)
        CV_RMSEs = np.zeros(k)

        for j in range(k):
            # Split up CV data sets
            these_CV_indices = CV_indices[j, :]
            these_CV_indices = these_CV_indices[~np.isnan(these_CV_indices)].astype(int)
            CV_valid_pigs = pigs_training[these_CV_indices]
            CV_valid_spec = spectra_4_mdl_training[these_CV_indices, :]
            CV_train_pigs = np.delete(pigs_training, these_CV_indices, axis=0)
            CV_train_spec = np.delete(spectra_4_mdl_training, these_CV_indices, axis=0)
            
            # standardize spectra for PCs
            CV_train_spec = (CV_train_spec - np.mean(CV_train_spec, axis=0)) / np.std(CV_train_spec, axis=0)
            CV_valid_spec = (CV_valid_spec - np.mean(CV_valid_spec, axis=0)) / np.std(CV_valid_spec, axis=0)

            # Manual PCA without centering using SVD
            U, S, VT = np.linalg.svd(CV_train_spec, full_matrices=False)

            CV_EOFs_train = VT[:max_components].T
            CV_AFs_train = U[:, :max_components] * S[:max_components]

            # Preallocate arrays to hold evaluation metrics
            n_val = len(CV_valid_pigs)
            percent_errors = np.zeros((n_val, CV_AFs_train.shape[1]))
            all_bias = np.zeros((n_val, CV_AFs_train.shape[1]))
            mean_percent_error = np.zeros(CV_AFs_train.shape[1])
            median_percent_error = np.zeros(CV_AFs_train.shape[1])
            bias = np.zeros(CV_AFs_train.shape[1])
            MAE = np.zeros(CV_AFs_train.shape[1])
            R2s = np.zeros(CV_AFs_train.shape[1])
            RMSEs = np.zeros(CV_AFs_train.shape[1])
            ensemble = np.zeros(CV_AFs_train.shape[1])
            pearson = np.zeros(CV_AFs_train.shape[1])

            # Loop over number of components used in model
            for l in range(1,CV_AFs_train.shape[1] + 1):
                # Multiple linear regression (MLR) using first l amplitude functions
                lin_model = LinearRegression()
                lin_model.fit(CV_AFs_train[:, :l], CV_train_pigs)

                # Intercept and coefficients
                this_alpha = lin_model.intercept_
                these_betas = lin_model.coef_

                # Map AF coefficients back to spectral domain (EOFs * weights)
                spec_betas = CV_EOFs_train[:, :l] @ these_betas

                # Apply model to validation spectra
                CV_modeled_pigs = CV_valid_spec @ spec_betas + this_alpha

                # Constrain results based on pft_index type
                if pft_index == 'pigment':
                    CV_modeled_pigs[CV_modeled_pigs < 0] = 0
                elif pft_index == 'compositions':
                    CV_modeled_pigs = np.clip(CV_modeled_pigs, 0, 1)
                elif pft_index == 'EOFs':
                    pass  # No constraints applied

                # Compute percent error
                percent_errors[:n_val, l-1] = ((CV_valid_pigs - CV_modeled_pigs) / CV_valid_pigs) * 100
                mean_percent_error[l-1] = np.mean(np.abs(percent_errors[:, l-1]))
                median_percent_error[l-1] = np.median(np.abs(percent_errors[:, l-1]))

                # Compute bias
                all_bias[:n_val, l-1] = CV_modeled_pigs - CV_valid_pigs
                bias[l-1] = np.mean(all_bias[:, l-1])
                MAE[l-1] = np.mean(np.abs(all_bias[:, l-1]))

                # Correlation and regression metrics
                reg = LinearRegression()
                reg.fit(CV_modeled_pigs.reshape(-1, 1), CV_valid_pigs)

                R2s[l-1] = reg.score(CV_modeled_pigs.reshape(-1, 1), CV_valid_pigs)
                RMSEs[l-1] = root_mean_squared_error(CV_modeled_pigs, CV_valid_pigs)
                pearson[l-1] = np.corrcoef(CV_modeled_pigs, CV_valid_pigs)[0, 1]

                # Ensemble score
                ensemble[l-1] = (1 - R2s[l-1] + RMSEs[l-1]) / 100

            # Select the best model based on the chosen metric
            if mdl_pick_metric == 'MAE':
                n_modes_to_use[j] = np.argmin(MAE) + 1 # account for python exclusive indexing
 
            # apply your optimized model and record the g.o.f. statistics for this k-th CV:
            X_train = CV_AFs_train[:, :n_modes_to_use[j]]
            y_train = CV_train_pigs

            lin_mdl = LinearRegression()
            lin_mdl.fit(X_train, y_train)

            alpha[j] = lin_mdl.intercept_  
            these_betas = lin_mdl.coef_

            # now turn model coefficients for AF's into coefficients for the combined
            # derivative spectra:
            betas[:, j] = CV_EOFs_train[:, :n_modes_to_use[j]] @ these_betas

            # 5-CV model validation
            CV_modeled_pigs = CV_valid_spec @ betas[:, j] + alpha[j]

            if pft_index == 'pigment':
                CV_modeled_pigs[CV_modeled_pigs < 0] = 0
            elif pft_index == 'compositions':
                CV_modeled_pigs = np.clip(CV_modeled_pigs, 0, 1)
            elif pft_index == 'EOFs':
                pass # No constraints applied

            # fit linear model to look at modeled vs observed
            CV_reg = LinearRegression()
            CV_reg.fit(CV_modeled_pigs.reshape(-1, 1), CV_valid_pigs)
            CV_R2s[j] = CV_reg.score(CV_modeled_pigs.reshape(-1, 1), CV_valid_pigs)
            CV_RMSEs[j] = root_mean_squared_error(CV_modeled_pigs, CV_valid_pigs)

        # so now you have k sets of optimized coefficients. grab the
        # average of them and validate against your original 25% validation
        # data set. 
        
        # Store mean/std of each set of k-fold CV betas 
        # (the model coefficients for the ith run of the n_permutations):
        mean_betas = np.mean(betas, axis=1)
        mean_alphas = np.mean(alpha)
        std_betas = np.std(betas, axis=1)
        std_alphas = np.std(alpha)

        # Compute standard deviation and mean across samples (i.e., along axis 0)
        spec_std = np.std(spectra_4_mdl_training, axis=0, ddof=0)  # MATLAB default is population std (ddof=0)
        spec_mean = np.mean(spectra_4_mdl_training, axis=0)

        # Unstandardize beta and alpha for model i
        mean_betas_nonstd[:, i] = mean_betas / spec_std
        mean_alphas_nonstd[i] = mean_alphas - np.sum(mean_betas * (spec_mean / spec_std))

        # Validate on the data you set aside previously for this ith run of the n_permutations using mean betas of this
        # permutation from cross-validation, and store g.o.f stats across permutations:
        modeled_pigs = spectra_4_mdl_validate @ mean_betas_nonstd[:,i] + mean_alphas_nonstd[i]

        if pft_index == 'pigment':
            modeled_pigs[modeled_pigs < 0] = 0
        elif pft_index == 'compositions':
            modeled_pigs = np.clip(modeled_pigs, 0, 1)
        elif pft_index == 'EOFs':
            pass # No constraints applied

        # Fit linear model
        model = LinearRegression().fit(modeled_pigs.reshape(-1, 1), pigs_validate)

        # Save R² and RMSE
        R2s_final[i] = model.score(modeled_pigs.reshape(-1, 1), pigs_validate)
        RMSEs_final[i] = np.sqrt(root_mean_squared_error(pigs_validate, model.predict(modeled_pigs.reshape(-1, 1))))

        # Avoid division by zero by replacing 0 with 1e-4
        pigs_validate_safe = np.where(pigs_validate == 0, 1e-4, pigs_validate)

        # Percent bias and errors
        pct_bias[i] = np.mean(((modeled_pigs - pigs_validate_safe) / pigs_validate_safe) * 100)
        pct_errors[i, :] = np.abs(((modeled_pigs - pigs_validate_safe) / pigs_validate_safe) * 100)
        med_pct_error[i] = np.median(pct_errors[i, :])
        avg_pct_error[i] = np.mean(pct_errors[i, :])

        # 95th percentile confidence interval
        sort_pct_errors = np.sort(pct_errors[i, :])
        CI_pct_error[i] = sort_pct_errors[int(np.ceil(0.95 * len(sort_pct_errors))) - 1]

        # Standard deviation of percent error
        std_pct_error[i] = np.std(pct_errors[i, :])

        # Mean absolute error
        mae_final[i] = np.mean(np.abs(modeled_pigs - pigs_validate))

        # Print progress
        #print(f"hey dude, im doing good. im on permutation # {i}")

    coefficients = mean_betas_nonstd
    intercepts = mean_alphas_nonstd

    # === Summary stats ===
    summary_gofs = [
        np.mean(R2s_final), np.std(R2s_final),
        np.mean(RMSEs_final), np.std(RMSEs_final),
        np.mean(avg_pct_error), np.std(avg_pct_error),
        np.mean(med_pct_error), np.std(med_pct_error),
        np.mean(pct_bias), np.std(pct_bias),
        np.mean(mae_final), np.std(mae_final)
    ]

    summary_gofs_df = pd.DataFrame([summary_gofs], columns=[
        'Mean_R2', 'SD_R2',
        'Mean_RMSE', 'SD_RMSE',
        'Mean_mean_pct_error', 'SD_mean_pct_error',
        'Mean_median_pct_error', 'SD_median_pct_error',
        'Mean_pct_bias', 'SD_pct_bias',
        'Mean_MAE', 'SD_MAE'
    ])

    # === All individual stats ===
    all_gofs = {
        'R2s': R2s_final,
        'RMSEs': RMSEs_final,
        'mean_pct_error': avg_pct_error,
        'median_pct_error': med_pct_error,
        'pct_bias': pct_bias,
        'all_pct_errors': pct_errors,
        'all_mae': mae_final
    }

    return coefficients, intercepts, summary_gofs_df, all_gofs