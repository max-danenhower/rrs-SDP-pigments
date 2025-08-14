import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
import pandas as pd

def rrsModelTrain(RrsD, hplc_i, pft_index, n_permutations, max_pcs, k, mdl_pick_metric):
    '''
    Run the PCA model.

    Parameters:
    -----------
    1. RrsD: numpy array (n_samples, n_wavelenghts)
        each spectrum/observation should be a row, and each RrsD(lambda) value a column
        (note that the model-training slices this data set so that each of the
        n_permutations blindly validates a model against an unknown validation
        data set)
    2. hplc_i: numpy array (n_samples)
        the corresponding values of the pft index you want to model with
        your input daph spectra. in other words, the order of observations/rows
        in pft should align with that in daph.
    3. pft_index : str
        specifies any constraints to apply to model outputs. 
        options: 
        (A)'pigment' --> model outputs are constrained to be >= 0 at
        each iteration of the model development. 
        (B) 'EOFs' --> model outputs are not constrained
        (C) 'compositions' --> model outputs are constrained to be >= 0 and <= 1
    4. n_permutations: int
        the numberof times you want to do the entire cross-validation exercise. 
    5. max_pcs: int
        the maximum number of spectral principal components that can
        be used in model training. the model will test/use all principal components up
        to and including this number
        NOTE: because the model optimization chops up your data set, max_pcs needs to 
        be less than or equal to: 0.75 * (1-1/k) * X; where k is input # 6, X is the number
        of observations in your data set
    6. k: int
        the number of actual model trainings to do for each of the n_permutations 
        cross-validation iterations. usually 5.
    7. mdl_pick_metric: str
        the gof statistic you want to optimize the model off of. 
        Options:
        (A) 'R2' --> picks mdls with maximum R2
        (B) 'RMSE' --> picks mdls with minimum RMSE
        (C) 'avg' --> minimum mean % error
        (D) 'med' --> minimum median % error
        (E) 'MAE' --> mean absolute error - see McKinna et al. 2021: https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2021JC017231 
    
    Returns:
    --------
    1. coefficients: numpy array (n_permutations, n_wavelengths)
        an array of model coefficients that is (n_permutations, size(RrsD,2))
        [which should be the number of wavelengths you've have derivative values for]

    2. intercepts: numpy array (n_permutations)
        a vector of model intercepts (should be reasonably close
        to 0) that is 1xn_permutations and contains the correspong intercept for
        each set of coefficients in coefficients array

    3. summary_gofs: pandas DataFrame
        a summary table of goodness of fit statistics across all
        n_permutations of model cross-validations

    4. all_gofs: dict
        a struct with each field a g.o.f statistic and each entry an
        array detailing all statistics from each of n_permutation
        cross-validations
    '''
    
    coefficients = None
    intercepts = None
    summary_gofs = None
    all_gofs = None

    # check for NaNs
    if np.isnan(hplc_i).any():
        print('hplc_i data has NaNs. remove them and try again plz')
        return coefficients, intercepts, summary_gofs, all_gofs
    elif np.isnan(RrsD).any():
        print('spectral data has NaNs. remove them and try again plz')
        return coefficients, intercepts, summary_gofs, all_gofs
    
    # check that RrsD and pft have the same number of rows
    if RrsD.shape[0] != hplc_i.shape[0]:
        print('RrsD and hplc_i must have the same number of rows')
        return coefficients, intercepts, summary_gofs, all_gofs
    
    max_components = max_pcs
    spectra_4_mdl = RrsD

    # set random number generator seed for reproducibility
    np.random.seed(42)

    # preallocate coefficients array
    mean_betas_nonstd = np.zeros((spectra_4_mdl.shape[1], n_permutations))
    mean_alphas_nonstd = np.zeros(n_permutations)

    # preallocate statistics/metrics arrays
    R2s_final = np.zeros(n_permutations)
    RMSEs_final = np.zeros(n_permutations)
    pct_bias = np.zeros(n_permutations)
    pct_errors = np.zeros((n_permutations,len(hplc_i)-int(len(hplc_i) * 0.75))) # 25% of the data for validation
    med_pct_error = np.zeros(n_permutations)
    avg_pct_error = np.zeros(n_permutations)
    CI_pct_error = np.zeros(n_permutations)
    std_pct_error = np.zeros(n_permutations)
    mae_final = np.zeros(n_permutations)

    for i in range(n_permutations):
        # Create broad training data (75%) and validation data (25%)

        training_indices = np.random.permutation(len(hplc_i))[:int(len(hplc_i) * 0.75)]

        pigs_training = hplc_i[training_indices]
        spectra_4_mdl_training = spectra_4_mdl[training_indices,:]

        # validation data
        pigs_validate = hplc_i
        pigs_validate = np.delete(pigs_validate,training_indices)
        spectra_4_mdl_validate = spectra_4_mdl
        spectra_4_mdl_validate = np.delete(spectra_4_mdl_validate, training_indices, axis=0)

        # get set up for k-fold cross validation

        pig_len = len(pigs_training)

        rand_ns = np.random.permutation(pig_len)

        CV_indices = np.full((k, int(np.ceil(len(pigs_training) / k))), np.nan)
        n_leftovers = pig_len % k
        counter_start = n_leftovers
        counter_end = n_leftovers + pig_len // k
        for j in range(k):
            CV_indices[j, :(pig_len // k)] = rand_ns[counter_start:counter_end]
            counter_start += pig_len // k
            counter_end += pig_len // k

        # add the leftovers to the CV_indices array, and put NaN's for the sets
        # where there's not enough leftovers to go around
        leftovers = rand_ns[:n_leftovers]
        na_array = np.full((k - len(leftovers)), np.nan)
        leftovers = np.concatenate([leftovers, na_array])

        CV_indices[:, CV_indices.shape[1]-1] = leftovers

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
            for l in range(1, CV_AFs_train.shape[1]+1):
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