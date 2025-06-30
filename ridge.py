import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score
from scipy.signal import savgol_filter
import os
from tqdm import tqdm
import warnings
import scipy.stats as stats
import csv
warnings.filterwarnings('ignore')

def process_data(df):
    """Preprocess data, including NA interpolation by dim and time period division"""
    print(f"Initial data shape: {df.shape}")
    
    # Check NA situation
    columns_to_check = ['milkweightlbs', 'cells', 'parity']
    na_counts = df[columns_to_check].isna().sum()
    print("\nNA counts in relevant columns:")
    print(na_counts[na_counts > 0])
    
    # Interpolate milkweightlbs and cells by dim
    for col in ['milkweightlbs', 'cells']:
        if df[col].isna().any():
            print(f"\nInterpolating {col} by dim...")
            df[col] = df.groupby('dim')[col].transform(lambda x: x.interpolate(method='linear'))
            
            # If there are still NAs, fill with the mean of that dim
            if df[col].isna().any():
                df[col] = df.groupby('dim')[col].transform(lambda x: x.fillna(x.mean()))
            
            # Check if there are still NAs
            remaining_na = df[col].isna().sum()
            if remaining_na > 0:
                print(f"Warning: {remaining_na} NA values remain in {col} after interpolation")
                df[col] = df[col].fillna(df[col].mean())
    
    # Process parity column as before (remove NAs)
    df = df.dropna(subset=['parity'])
    print(f"\nShape after handling NA: {df.shape}")
    
    # Reclassify parity
    df['parity'] = df['parity'].apply(lambda x: '2+' if x > 2 else str(x))
    
    # Only keep samples with disease 0 and 1
    df = df[df['disease'].isin([0, 1])].copy()
    print(f"\nShape after selecting disease 0 and 1: {df.shape}")
    
    # Create time periods for disease group (disease=1)
    def create_time_group(days):
        if days > 10:
            return '>10'
        elif 8 <= days <= 10:
            return '10-8'
        elif 6 <= days <= 7:
            return '7-6'
        elif 4 <= days <= 5:
            return '5-4'
        elif days == 3:
            return '3'
        elif days == 2:
            return '2'
        elif days == 1:
            return '1'
        else:
            return '0'
    
    # Add time group column
    df['time_group'] = None
    disease_mask = df['disease'] == 1
    df.loc[disease_mask, 'time_group'] = df.loc[disease_mask, 'disease_in'].apply(create_time_group)
    df.loc[~disease_mask, 'time_group'] = 'healthy'
    
    print("\nSample counts by time group:")
    print(df['time_group'].value_counts())
    
    # Check final data
    print(f"\nFinal data shape: {df.shape}")
    print("\nFinal NA counts:")
    print(df[columns_to_check].isna().sum())
    
    return df

def get_spectral_data(df, type='original'):
    """Get spectral data
    type: 'original', 'derivative', or 'rmR4'
    """
    # Get spectral columns (assuming non-spectral columns are known)
    non_spectral_cols = ['disease_in', 'disease', 'day_group', 'milkweightlbs', 
                        'cells', 'parity', 'Unnamed: 0', 'index']  # Add additional non-spectral columns
    
    # Get all numeric column names (spectral wavelengths)
    spectral_cols = [col for col in df.columns if col not in non_spectral_cols]
    
    # Ensure all column names can be converted to float
    spectral_cols = [col for col in spectral_cols if col.replace('.', '').isdigit()]
    
    # Convert column names to numeric values for range selection
    wavelengths = [float(col) for col in spectral_cols]
    
    # Select columns in the range 1000-3000, excluding 1580-1700 and 1800-2800
    valid_cols = [col for col, wave in zip(spectral_cols, wavelengths)
                 if 1000 <= wave <= 3000 and not (1580 <= wave <= 1700) and not (1800 <= wave <= 2800)]
    
    if type == 'original':
        return df[valid_cols]
    elif type == 'derivative':
        # Calculate first derivative
        spectra = df[valid_cols].values
        derivatives = savgol_filter(spectra, window_length=7, polyorder=2, deriv=1, axis=1)
        return pd.DataFrame(derivatives, columns=valid_cols, index=df.index)
    elif type == 'rmR4':
        # Remove 1800-2800 region from original
        # rmR4_cols = [col for col, wave in zip(valid_cols, map(float, valid_cols))
        #             if wave < 1800 or wave > 2800]
        rmR4_cols = [col for col, wave in zip(spectral_cols, wavelengths)
                 if 1000 <= wave <= 3000 and not (1580 <= wave <= 1700)]
        return df[rmR4_cols]
    else:
        raise ValueError(f"Unknown spectral type: {type}")

def calculate_derivatives(spectra):
    """Calculate first derivative"""
    return pd.DataFrame(
        savgol_filter(spectra, window_length=7, polyorder=2, deriv=1),
        columns=spectra.columns,
        index=spectra.index
    )

def prepare_features(df, spectral_data, feature_type='spc'):
    """Prepare feature data"""
    
    # Convert to numpy array
    spectral_array = spectral_data.values
    
    # Scale each spectrum to [0,1] range
    for i in range(spectral_array.shape[0]):
        spectrum = spectral_array[i]
        min_val = np.min(spectrum)
        max_val = np.max(spectrum)
        if max_val > min_val:
            spectral_array[i] = (spectrum - min_val) / (max_val - min_val)
    
    # Prepare data based on feature_type
    if feature_type == 'spc':
        X = spectral_array
    else:
        # Prepare individual variables
        scaler = StandardScaler()
        encoder = OneHotEncoder(sparse=False)
        
        features = []
        if 'my' in feature_type:
            milk_weight = df[['milkweightlbs']].values
            min_val = np.min(milk_weight)
            max_val = np.max(milk_weight)
            if max_val > min_val:
                milk_weight = (milk_weight - min_val) / (max_val - min_val)
            features.append(milk_weight)
            
        if 'scc' in feature_type:
            scc = df[['cells']].values
            min_val = np.min(scc)
            max_val = np.max(scc)
            if max_val > min_val:
                scc = (scc - min_val) / (max_val - min_val)
            features.append(scc)
            
        if 'parity' in feature_type:
            parity = encoder.fit_transform(df[['parity']])
            features.append(parity)
            
        # Combine features
        X = np.hstack([spectral_array] + features)
    
    return X

def calculate_metrics(y_true, y_pred_proba):
    """Calculate performance metrics"""
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Check if there is only one class
    unique_classes = np.unique(y_true)
    if len(unique_classes) == 1:
        print(f"Warning: Only one class present ({unique_classes[0]}). Skipping metrics calculation.")
        return {
            'auc': np.nan,
            'acc': np.nan,
            'sen': np.nan,
            'spc': np.nan
        }
    
    try:
        return {
            'auc': roc_auc_score(y_true, y_pred_proba),
            'acc': accuracy_score(y_true, y_pred),
            'sen': recall_score(y_true, y_pred),
            'spc': recall_score(y_true, y_pred, pos_label=0)
        }
    except Exception as e:
        print(f"Warning: Error calculating metrics: {str(e)}")
        return {
            'auc': np.nan,
            'acc': np.nan,
            'sen': np.nan,
            'spc': np.nan
        }

def calculate_ci(values, confidence=0.95):
    """Calculate confidence interval"""
    mean = np.mean(values)
    se = stats.sem(values)
    ci = stats.t.interval(confidence, len(values)-1, loc=mean, scale=se)
    return mean, ci[0], ci[1]

def sensitivity_score(y_true, y_pred):
    """Calculate sensitivity (same as recall)"""
    return recall_score(y_true, y_pred)

def specificity_score(y_true, y_pred):
    """Calculate specificity"""
    # Specificity = TN / (TN + FP) = recall_score for negative class
    return recall_score(y_true, y_pred, pos_label=0)

def calculate_coefficient_importance(model, X):
    """Calculate importance of coefficients in Ridge model"""
    return np.abs(model.coef_[0])

def double_cv(X, y, alpha_range, cv1=5, cv2=5, n_repeats_cv2=5):
    """
    Perform double cross-validation for Ridge model.
    
    Args:
        X (np.ndarray): Feature matrix
        y (np.ndarray): Target labels
        alpha_range (list): Range of alpha values to test
        cv1 (int): Number of inner CV folds
        cv2 (int): Number of outer CV folds
        n_repeats_cv2 (int): Number of repeats for outer CV
        
    Returns:
        tuple: Test predictions, true labels, feature importance, inner CV performance, and detailed CV metrics
    """
    try:
        test_predictions = []
        test_true = []
        feature_importance = []
        best_alpha_list = []
        detailed_cv_metrics = []  # Store detailed CV metrics
        
        # Initialize storage for inner CV performances
        inner_cv_performances = {
            'train': {'auc': [], 'acc': [], 'sen': [], 'spc': []},
            'val': {'auc': [], 'acc': [], 'sen': [], 'spc': []}
        }
        
        # Repeat outer CV
        for repeat in range(n_repeats_cv2):
            # Outer CV
            outer_cv = KFold(n_splits=cv2, shuffle=True)
            
            for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X)):
                X_rest, X_test = X[train_idx], X[test_idx]
                y_rest, y_test = y[train_idx], y[test_idx]
                
                # Check for sufficient samples and classes
                if len(np.unique(y_rest)) < 2 or len(np.unique(y_test)) < 2:
                    print("Warning: Insufficient classes in train or test set")
                    continue
                
                # Inner CV
                best_alpha = alpha_range[0]  # Default to first value
                best_score = -np.inf
                
                inner_cv = KFold(n_splits=cv1, shuffle=True)
                for alpha in alpha_range:
                    val_scores = []
                    train_scores = []
                    
                    for inner_fold_idx, (train_inner_idx, val_idx) in enumerate(inner_cv.split(X_rest)):
                        X_train, X_val = X_rest[train_inner_idx], X_rest[val_idx]
                        y_train, y_val = y_rest[train_inner_idx], y_rest[val_idx]
                        
                        # Train model
                        model = RidgeClassifier(alpha=alpha)
                        model.fit(X_train, y_train)
                        
                        # Calculate training performance
                        y_train_pred = model.predict(X_train)
                        train_metrics = calculate_metrics(y_train, y_train_pred)
                        train_scores.append(train_metrics)
                        
                        # Calculate validation performance
                        y_val_pred = model.predict(X_val)
                        val_metrics = calculate_metrics(y_val, y_val_pred)
                        val_scores.append(val_metrics)
                        
                        # Store detailed metrics
                        detailed_cv_metrics.append({
                            'repeat': repeat + 1,
                            'outer_fold': fold_idx + 1,
                            'inner_fold': inner_fold_idx + 1,
                            'alpha': alpha,
                            'train_auc': train_metrics['auc'],
                            'train_acc': train_metrics['acc'],
                            'train_sen': train_metrics['sen'],
                            'train_spc': train_metrics['spc'],
                            'val_auc': val_metrics['auc'],
                            'val_acc': val_metrics['acc'],
                            'val_sen': val_metrics['sen'],
                            'val_spc': val_metrics['spc']
                        })
                    
                    # Calculate average performance
                    mean_train_metrics = {k: np.mean([s[k] for s in train_scores]) for k in train_metrics}
                    mean_val_metrics = {k: np.mean([s[k] for s in val_scores]) for k in val_metrics}
                    
                    # Store inner CV performance
                    for metric in ['auc', 'acc', 'sen', 'spc']:
                        inner_cv_performances['train'][metric].append(mean_train_metrics[metric])
                        inner_cv_performances['val'][metric].append(mean_val_metrics[metric])
                    
                    # Use validation AUC to select best alpha
                    mean_val_auc = mean_val_metrics['auc']
                    if mean_val_auc > best_score:
                        best_score = mean_val_auc
                        best_alpha = alpha
                
                # Train final model with best parameters
                final_model = RidgeClassifier(alpha=best_alpha)
                final_model.fit(X_rest, y_rest)
                
                # Calculate feature importance
                try:
                    importance = calculate_coefficient_importance(final_model, X_rest)
                    feature_importance.append(importance)
                except Exception as e:
                    print(f"Warning: Error calculating feature importance: {str(e)}")
                    feature_importance.append(None)
                
                # Predict test set
                y_test_pred = final_model.predict(X_test)
                test_predictions.extend(y_test_pred)
                test_true.extend(y_test)
                best_alpha_list.append({
                    'fold': fold_idx + 1,
                    'best_alpha': best_alpha,
                    'best_val_auc': best_score
                })
        
        # Calculate average inner CV performance (using nanmean for NA values)
        avg_inner_cv_performance = {
            'train': {k: np.nanmean(v) if v else np.nan for k, v in inner_cv_performances['train'].items()},
            'val': {k: np.nanmean(v) if v else np.nan for k, v in inner_cv_performances['val'].items()}
        }
        
        return np.array(test_predictions), np.array(test_true), feature_importance, avg_inner_cv_performance, best_alpha_list, detailed_cv_metrics
    except Exception as e:
        print(f"Error in double_cv: {str(e)}")
        return np.array([]), np.array([]), [], {'train': {}, 'val': {}}, [], []

def permutation_test(orig_metrics, X, y, alpha_range, n_permutations=1000, **kwargs):
    """Perform permutation test"""
    # First downsample the original data
    class1_idx = np.where(y == 0)[0]
    class2_idx = np.where(y == 1)[0]
    min_samples = min(len(class1_idx), len(class2_idx))
    
    # Downsample to minority class size
    if len(class1_idx) > min_samples:
        class1_idx = np.random.choice(class1_idx, size=min_samples, replace=False)
    if len(class2_idx) > min_samples:
        class2_idx = np.random.choice(class2_idx, size=min_samples, replace=False)
    
    # Combine selected samples
    selected_idx = np.concatenate([class1_idx, class2_idx])
    X_balanced = X[selected_idx]
    y_balanced = y[selected_idx]
    
    # Permutation test
    null_metrics = []
    for _ in tqdm(range(n_permutations), desc="Permutation test"):
        # Permute labels
        y_perm = np.random.permutation(y_balanced)
        
        # Perform double_cv
        print(f"samples in X_balanced: {len(X_balanced)}")
        perm_pred, perm_true, _, _ = double_cv(
            X_balanced, 
            y_perm,
            alpha_range=alpha_range,
            **kwargs
        )
        
        # Only calculate metrics if predictions are not empty
        if len(perm_pred) > 0:
            metrics = calculate_metrics(perm_true, perm_pred)
            null_metrics.append(metrics)
    
    # Return empty results if no valid permutation results
    if not null_metrics:
        print("Warning: No valid permutation results")
        return pd.DataFrame(), {
            'auc': np.nan,
            'acc': np.nan,
            'sen': np.nan,
            'spc': np.nan
        }
    
    # Calculate p-values
    null_metrics = pd.DataFrame(null_metrics)
    p_values = {
        metric: (np.sum(null_metrics[metric] >= orig_metrics[metric]) + 1) / (n_permutations + 1)
        for metric in ['auc', 'acc', 'sen', 'spc']
    }
    
    return null_metrics, p_values

def prepare_balanced_data(df, time_group):
    """
    Prepare balanced dataset for a specific time group, ensuring dim matching between healthy and disease groups.
    
    Args:
        df (pd.DataFrame): Input dataframe
        time_group (str): Time group to prepare data for
        
    Returns:
        tuple: Balanced dataframe and labels
    """
    # Get disease group samples
    disease_samples = df[df['time_group'] == time_group]
    
    # Get unique dim values from disease group
    disease_dims = disease_samples['dim'].unique()
    
    # For each dim in disease group, get matching healthy samples
    healthy_samples = []
    for dim in disease_dims:
        # Get healthy samples with this dim
        dim_healthy = df[(df['time_group'] == 'healthy') & (df['dim'] == dim)]
        if len(dim_healthy) > 0:
            # Randomly sample one healthy sample for this dim
            healthy_samples.append(dim_healthy.sample(n=1, random_state=42))
    
    # Combine all healthy samples
    if healthy_samples:
        healthy_samples = pd.concat(healthy_samples)
    else:
        healthy_samples = pd.DataFrame(columns=df.columns)
    
    # Combine samples
    balanced_df = pd.concat([disease_samples, healthy_samples])
    
    # Prepare labels (0: healthy, 1: disease)
    y = (balanced_df['disease'] == 1).astype(int)
    
    return balanced_df, y

def main(input_file, output_dir, repeat=50, cv2=5, cv1=5, n_repeats_cv2=5, run_permutation=True):
    """Main function"""
    print(f"Starting analysis with input file: {input_file}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory created: {output_dir}")
    
    # Read data
    print("Reading input data...")
    df = pd.read_csv(input_file)
    df = process_data(df)
    print(f"Data loaded and processed. Shape: {df.shape}")
    
    # Prepare result storage
    all_results = []  # Store results for all time groups and repeats
    all_importance = []  # Store feature importance for all time groups and repeats
    all_permutations = []  # Store permutation test results for all time groups
    all_p_values = []  # Store p-values for all time groups
    skipped_cases = []  # Store skipped cases
    all_best_alphas = []  # Store best alpha values
    all_detailed_cv_metrics = []  # Store detailed CV metrics
    
    # Feature combinations
    feature_types = [
        'my+scc+dim+parity',
        'my+scc+dim',
        'my+scc',
        'totalfa+lactose+protein',
        'spc',
        'spc_de',
        'spc_rmR4',
        'my+scc+dim+parity+totalfa+lactose+protein',
        'my+scc+dim+parity+totalfa+lactose+protein+spc',
        'my+scc+dim+parity+totalfa+lactose+protein+spc_de'
    ]

    # Add _max20 suffix versions for features containing spc
    spc_features = [f for f in feature_types if 'spc' in f]
    feature_types.extend([f + '_max20' for f in spc_features])

    # Time groups
    time_groups = ['>10', '10-8', '7-6', '5-4', '3', '2', '1', '0']
    
    for time_group in time_groups:
        print(f"\nProcessing time group: {time_group}")
        
        # Prepare balanced dataset
        balanced_df, y = prepare_balanced_data(df, time_group)
        n_samples = len(balanced_df)
        
        print(f"Number of samples in time group {time_group}: {n_samples}")
        
        if n_samples < 10:  # Skip if insufficient samples
            msg = f"Skipping time group {time_group}: insufficient samples ({n_samples})"
            print(msg)
            skipped_cases.append({
                'time_group': time_group,
                'reason': msg
            })
            continue
            
        for feature_type in feature_types:
            print(f"\nProcessing feature type: {feature_type}")
            
            # Handle feature type name, remove _max20 suffix for feature extraction
            base_feature_type = feature_type.replace('_max20', '')
            
            # Select alpha range based on feature type
            if 'spc' in base_feature_type:
                if '_max20' in feature_type:
                    alpha_range = [0.1, 1, 10, 100, 1000, 10000]
                    print("Using extended alpha_range for spectral data:", alpha_range)
                else:
                    alpha_range = [0.1, 1, 10, 100, 1000]
                    print("Using extended alpha_range for spectral data:", alpha_range)
            else:
                alpha_range = [0.1, 1, 10, 100]
                print("Using standard alpha_range:", alpha_range)
            
            # Store metrics for each repeat
            repeat_metrics = {
                'outer_auc': [], 'outer_acc': [], 'outer_sen': [], 'outer_spc': [],
                'inner_train_auc': [], 'inner_train_acc': [], 'inner_train_sen': [], 'inner_train_spc': [],
                'inner_val_auc': [], 'inner_val_acc': [], 'inner_val_sen': [], 'inner_val_spc': []
            }
            
            # Prepare complete dataset for permutation test
            balanced_data_full = balanced_df.copy()
            
            if feature_type == 'spc_de' or '_max20' in feature_type and 'spc_de' in base_feature_type:
                spectral_type = 'derivative'
                base_feature = 'spc'
            elif feature_type == 'spc_rmR4' or '_max20' in feature_type and 'spc_rmR4' in base_feature_type:
                spectral_type = 'rmR4'
                base_feature = 'spc'
            else:
                spectral_type = 'original'
                base_feature = base_feature_type
            
            balanced_spectral_full = get_spectral_data(balanced_data_full, type=spectral_type)
            X_full = prepare_features(balanced_data_full, balanced_spectral_full, feature_type=base_feature)
            y_full = balanced_data_full['disease'].values
            
            for i in range(repeat):
                print(f"\nRepeat {i+1}/{repeat}")
                
                # Downsample to smaller class size
                min_samples = min(len(balanced_df[balanced_df['time_group'] == time_group]), len(balanced_df[balanced_df['time_group'] == 'healthy']))
                if len(balanced_df[balanced_df['time_group'] == time_group]) > min_samples:
                    sampled_indices1 = np.random.choice(balanced_df[balanced_df['time_group'] == time_group].index, size=min_samples, replace=False)
                    balanced_samples1 = balanced_df.loc[sampled_indices1]
                else:
                    balanced_samples1 = balanced_df[balanced_df['time_group'] == time_group]
                
                if len(balanced_df[balanced_df['time_group'] == 'healthy']) > min_samples:
                    sampled_indices2 = np.random.choice(balanced_df[balanced_df['time_group'] == 'healthy'].index, size=min_samples, replace=False)
                    balanced_samples2 = balanced_df.loc[sampled_indices2]
                else:
                    balanced_samples2 = balanced_df[balanced_df['time_group'] == 'healthy']
                
                # Combine data and prepare labels
                balanced_data = pd.concat([balanced_samples1, balanced_samples2]).reset_index(drop=True)
                balanced_data['disease'] = (balanced_data['time_group'] == time_group).astype(int)
                
                # Prepare features
                if feature_type == 'spc_de' or '_max20' in feature_type and 'spc_de' in base_feature_type:
                    spectral_type = 'derivative'
                    base_feature = 'spc'
                elif feature_type == 'spc_rmR4' or '_max20' in feature_type and 'spc_rmR4' in base_feature_type:
                    spectral_type = 'rmR4'
                    base_feature = 'spc'
                else:
                    spectral_type = 'original'
                    base_feature = base_feature_type
                
                balanced_spectral = get_spectral_data(balanced_data, type=spectral_type)
                
                X = prepare_features(balanced_data, balanced_spectral, feature_type=base_feature_type)
                y = balanced_data['disease'].values
                
                # Perform double cross-validation
                pred, true, importance, inner_cv_perf, best_alphas, detailed_cv_metrics = double_cv(
                    X, y, 
                    alpha_range=alpha_range,
                    cv1=cv1, 
                    cv2=cv2,
                    n_repeats_cv2=n_repeats_cv2
                )
                
                metrics = calculate_metrics(true, pred)
                
                # Store results for this repeat
                for key in repeat_metrics:
                    if key.startswith('outer_'):
                        metric_name = key[6:]
                        repeat_metrics[key].append(metrics[metric_name])
                    elif key.startswith('inner_'):
                        phase, metric_name = key[6:].split('_', 1)
                        repeat_metrics[key].append(inner_cv_perf[phase][metric_name])
                
                # Store feature importance (only for original spectral features)
                if feature_type == 'spc' and importance is not None:
                    wavelengths = [float(col) for col in balanced_spectral.columns]
                    for fold_idx, fold_importance in enumerate(importance):
                        if fold_importance is not None:
                            for wave_idx, wave in enumerate(wavelengths):
                                all_importance.append({
                                    'time_group': time_group,
                                    'feature': feature_type,
                                    'repeat': i+1,
                                    'fold': fold_idx + 1,
                                    'wavenumber': wave,
                                    'importance': fold_importance[wave_idx]
                                })
                
                # Record best alpha values
                for alpha_info in best_alphas:
                    all_best_alphas.append({
                        'time_group': time_group,
                        'feature_type': feature_type,  # Use original feature type name (with _max20 suffix)
                        'repeat': i + 1,
                        'fold': alpha_info['fold'],
                        'best_alpha': alpha_info['best_alpha'],
                        'best_val_auc': alpha_info['best_val_auc']
                    })
                
                # Store detailed CV metrics
                for metric in detailed_cv_metrics:
                    metric['time_group'] = time_group
                    metric['feature_type'] = feature_type
                    metric['repeat'] = i + 1
                    all_detailed_cv_metrics.append(metric)
            
            # Calculate mean and CI for each metric
            final_metrics = {}
            for key in repeat_metrics:
                values = np.array(repeat_metrics[key])
                mean, ci_low, ci_up = calculate_ci(values)
                final_metrics[f'{key}_mean'] = mean
                final_metrics[f'{key}_ci_low'] = ci_low
                final_metrics[f'{key}_ci_up'] = ci_up
            
            # Store results
            all_results.append({
                'time_group': time_group,
                'feature': feature_type,
                **final_metrics
            })
            
            # Perform permutation test
            if run_permutation:
                print(f"\nPerforming permutation test for {feature_type}...")
                null_metrics, p_values = permutation_test(
                    metrics,
                    X_full, y_full,
                    alpha_range=alpha_range,
                    n_permutations=1000,
                    cv1=cv1,
                    cv2=cv2
                )
                
                # Store permutation results
                for _, row in null_metrics.iterrows():
                    all_permutations.append({
                        'time_group': time_group,
                        'feature': feature_type,
                        'null_auc': row['auc'],
                        'null_acc': row['acc'],
                        'null_sen': row['sen'],
                        'null_spc': row['spc']
                    })
                
                # Store p-values
                p_values.update({
                    'time_group': time_group,
                    'feature': feature_type
                })
                all_p_values.append(p_values)
    
    # Save results
    print("\nSaving results...")
    
    # Convert to DataFrame and ensure correct column names
    if all_results:
        results_df = pd.DataFrame(all_results)
        # Add 't_' prefix to time_group
        results_df['time_group'] = results_df['time_group'].apply(lambda x: f"t_{x}" if x != 'healthy' else x)
    else:
        results_df = pd.DataFrame(columns=['time_group', 'feature'])
        
    if all_importance:
        importance_df = pd.DataFrame(all_importance)
        # Add 't_' prefix to time_group
        importance_df['time_group'] = importance_df['time_group'].apply(lambda x: f"t_{x}" if x != 'healthy' else x)
    else:
        importance_df = pd.DataFrame(columns=['time_group', 'feature', 'repeat', 'fold', 'wavenumber', 'importance'])
        
    if all_permutations:
        permutation_df = pd.DataFrame(all_permutations)
        # Add 't_' prefix to time_group
        permutation_df['time_group'] = permutation_df['time_group'].apply(lambda x: f"t_{x}" if x != 'healthy' else x)
    else:
        permutation_df = pd.DataFrame(columns=['time_group', 'feature', 'null_auc', 'null_acc', 'null_sen', 'null_spc'])
        
    if all_p_values:
        p_values_df = pd.DataFrame(all_p_values)
        # Add 't_' prefix to time_group
        p_values_df['time_group'] = p_values_df['time_group'].apply(lambda x: f"t_{x}" if x != 'healthy' else x)
    else:
        p_values_df = pd.DataFrame(columns=['time_group', 'feature', 'auc', 'acc', 'sen', 'spc'])
    
    if skipped_cases:
        skipped_df = pd.DataFrame(skipped_cases)
    else:
        skipped_df = pd.DataFrame(columns=['time_group', 'reason'])
    
    if all_best_alphas:
        best_alphas_df = pd.DataFrame(all_best_alphas)
        best_alphas_df.to_csv(os.path.join(output_dir, 'best_alphas.csv'), index=False)
        print(f"\nSaved best alpha values to best_alphas.csv")
        
        # Calculate and save summary statistics
        summary_stats = best_alphas_df.groupby(['time_group', 'feature_type'])['best_alpha'].agg([
            'mean', 'std', 'min', 'max', 'median'
        ]).round(2)
        summary_stats.to_csv(os.path.join(output_dir, 'best_alphas_summary.csv'))
        print(f"Saved best alpha summary statistics to best_alphas_summary.csv")
    
    if all_detailed_cv_metrics:
        detailed_cv_df = pd.DataFrame(all_detailed_cv_metrics)
        detailed_cv_df.to_csv(os.path.join(output_dir, 'detailed_cv_metrics.csv'), index=False)
        print(f"Saved detailed CV metrics to detailed_cv_metrics.csv")
    
    # Create output directory (if it doesn't exist)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save all results to single files
    if not results_df.empty:
        results_df.to_csv(os.path.join(output_dir, 'pre.csv'), index=False)
        print(f"Saved performance results to pre.csv")
        
    if not importance_df.empty:
        importance_df.to_csv(os.path.join(output_dir, 'imp.csv'), index=False)
        print(f"Saved importance results to imp.csv")
    
    if run_permutation:
        if not permutation_df.empty:
            permutation_df.to_csv(os.path.join(output_dir, 'permu.csv'), index=False)
            print(f"Saved permutation results to permu.csv")
        
        if not p_values_df.empty:
            p_values_df.to_csv(os.path.join(output_dir, 'p_values.csv'), index=False)
            print(f"Saved p-values to p_values.csv")
    
    if not skipped_df.empty:
        skipped_df.to_csv(os.path.join(output_dir, 'skipped.csv'), index=False)
        print(f"Saved skipped cases to skipped.csv")
    
    print("\nResults saved successfully!")
    
    # Print basic statistics
    print("\nSummary of results:")
    if not results_df.empty:
        print(f"\nNumber of time groups: {results_df['time_group'].nunique()}")
        print(f"Number of features: {results_df['feature'].nunique()}")
        print("\nSamples per time group:")
        print(results_df.groupby('time_group').size())
    
    if not importance_df.empty:
        print(f"\nNumber of importance scores: {len(importance_df)}")
        print(f"Number of wavelengths: {importance_df['wavenumber'].nunique()}")
    
    if run_permutation and not permutation_df.empty:
        print(f"\nNumber of permutation tests: {len(permutation_df)}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', required=True, help='Input CSV file path')
    parser.add_argument('--output_dir', required=True, help='Output directory path')
    parser.add_argument('--repeat', type=int, default=50, help='Number of downsample repeats')
    parser.add_argument('--cv2', type=int, default=5, help='Number of outer CV folds')
    parser.add_argument('--cv1', type=int, default=5, help='Number of inner CV folds')
    parser.add_argument('--n_repeats_cv2', type=int, default=5, help='Number of repeats for outer CV')
    parser.add_argument('--run_permutation', action='store_true', help='Run permutation test')
    
    args = parser.parse_args()
    main(**vars(args))