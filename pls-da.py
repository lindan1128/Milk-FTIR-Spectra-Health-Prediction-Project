import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cross_decomposition import PLSRegression
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
    """
    Process the input data by converting time_group values to numeric format.
    
    Args:
        df (pd.DataFrame): Input dataframe containing time_group column
        
    Returns:
        pd.DataFrame: Processed dataframe with numeric time_group values
    """
    print(f"Initial data shape: {df.shape}")
    
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
    
    df['time_group'] = df['time_group'].replace('healthy', '0')
    df['time_group'] = df['time_group'].astype(float)
    return df

def get_spectral_data(df, type='original'):
    """
    Extract spectral data from the dataframe based on specified type.
    
    Args:
        df (pd.DataFrame): Input dataframe
        type (str): Type of spectral data to extract ('original', 'derivative', or 'rmR4')
        
    Returns:
        pd.DataFrame: Extracted spectral data
    """
    # Get spectral columns (assuming non-spectral columns are known)
    non_spectral_cols = ['disease_in', 'disease', 'day_group', 'milkweightlbs', 
                        'cells', 'parity', 'Unnamed: 0', 'index']  # 添加额外的非光谱列
    
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
        raise ValueError(f"Invalid spectral type: {type}")

def calculate_derivatives(spectra):
    """
    Calculate first and second derivatives of spectral data.
    
    Args:
        spectral_data (pd.DataFrame): Original spectral data
        
    Returns:
        tuple: First and second derivatives of the spectral data
    """
    # First derivative
    first_derivative = pd.DataFrame(index=spectra.index)
    for i in range(1, len(spectra.columns)-1):
        first_derivative[f'derivative_{i}'] = (spectra.iloc[:, i+1] - spectra.iloc[:, i-1]) / 2
    
    # Second derivative
    second_derivative = pd.DataFrame(index=spectra.index)
    for i in range(2, len(spectra.columns)-2):
        second_derivative[f'derivative2_{i}'] = (spectra.iloc[:, i+2] - 2*spectra.iloc[:, i] + spectra.iloc[:, i-2]) / 4
    
    return first_derivative, second_derivative

def prepare_features(df, spectral_data, feature_type='spc'):
    """
    Prepare feature matrix based on specified feature type.
    
    Args:
        df (pd.DataFrame): Input dataframe
        spectral_data (pd.DataFrame): Spectral data
        feature_type (str): Type of features to include
        
    Returns:
        np.ndarray: Prepared feature matrix
    """
    if feature_type == 'spc':
        return spectral_data.values
    elif feature_type == 'spc+my':
        return np.column_stack([spectral_data.values, df['my'].values.reshape(-1, 1)])
    elif feature_type == 'spc+scc':
        return np.column_stack([spectral_data.values, df['scc'].values.reshape(-1, 1)])
    elif feature_type == 'spc+parity':
        return np.column_stack([spectral_data.values, df['parity'].values.reshape(-1, 1)])
    elif feature_type == 'spc+my+scc+parity':
        return np.column_stack([
            spectral_data.values,
            df['my'].values.reshape(-1, 1),
            df['scc'].values.reshape(-1, 1),
            df['parity'].values.reshape(-1, 1)
        ])
    else:
        raise ValueError(f"Invalid feature type: {feature_type}")

def calculate_metrics(y_true, y_pred):
    """
    Calculate performance metrics for binary classification.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted probabilities
        
    Returns:
        dict: Dictionary containing performance metrics
    """
    # Convert probabilities to binary predictions
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    # Calculate metrics
    auc = roc_auc_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred_binary)
    sen = sensitivity_score(y_true, y_pred_binary)
    spc = specificity_score(y_true, y_pred_binary)
    
    return {
        'auc': auc,
        'acc': acc,
        'sen': sen,
        'spc': spc
    }

def calculate_ci(values, confidence=0.95):
    """
    Calculate mean and confidence interval for a set of values.
    
    Args:
        values (np.ndarray): Array of values
        confidence (float): Confidence level (default: 0.95)
        
    Returns:
        tuple: Mean, lower CI, and upper CI
    """
    mean = np.mean(values)
    std_err = np.std(values) / np.sqrt(len(values))
    ci = std_err * 1.96  # 95% confidence interval
    return mean, mean - ci, mean + ci

def calculate_vip(model, X):
    """
    Calculate VIP (Variable Importance in Projection) scores for PLS model.
    
    Args:
        model (PLSRegression): Fitted PLS model
        X (np.ndarray): Feature matrix
        
    Returns:
        np.ndarray: VIP scores for each feature
    """
    t = model.x_scores_
    w = model.x_weights_
    q = model.y_loadings_
    p, h = w.shape
    
    vips = np.zeros((p,))
    s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
    total_s = np.sum(s)
    
    for i in range(p):
        weight = np.array([(w[i,j] / np.linalg.norm(w[:,j]))**2 for j in range(h)])
        vips[i] = np.sqrt(p * (weight @ s) / total_s)
    
    return vips

def sensitivity_score(y_true, y_pred):
    """
    Calculate sensitivity (true positive rate).
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        
    Returns:
        float: Sensitivity score
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp / (tp + fn) if (tp + fn) > 0 else 0

def specificity_score(y_true, y_pred):
    """
    Calculate specificity (true negative rate).
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        
    Returns:
        float: Specificity score
    """
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tn / (tn + fp) if (tn + fp) > 0 else 0

def double_cv(X, y, n_components_range, cv1=5, cv2=5, n_repeats_cv2=5):
    """
    Perform double cross-validation for PLS-DA model.
    
    Args:
        X (np.ndarray): Feature matrix
        y (np.ndarray): Target labels
        n_components_range (list): Range of components to test
        cv1 (int): Number of inner CV folds
        cv2 (int): Number of outer CV folds
        n_repeats_cv2 (int): Number of repeats for outer CV
        
    Returns:
        tuple: Test predictions, true labels, feature importance, and inner CV performance
    """
    try:
        test_predictions = []
        test_true = []
        feature_importance = []
        best_n_components_list = []
        
        # Initialize storage for inner CV performances
        inner_cv_performances = {
            'train': {'auc': [], 'acc': [], 'sen': [], 'spc': []},
            'val': {'auc': [], 'acc': [], 'sen': [], 'spc': []}
        }
        
        # Repeat outer CV
        for repeat in range(n_repeats_cv2):
            # Outer CV
            outer_cv = KFold(n_splits=cv2, shuffle=True)
            
            for train_idx, test_idx in outer_cv.split(X):
                X_rest, X_test = X[train_idx], X[test_idx]
                y_rest, y_test = y[train_idx], y[test_idx]
                
                # Check for sufficient samples and classes
                if len(np.unique(y_rest)) < 2 or len(np.unique(y_test)) < 2:
                    print("Warning: Insufficient classes in train or test set")
                    continue
                
                # Inner CV
                best_n_components = n_components_range[0]  # Default to first value
                best_score = -np.inf
                
                inner_cv = KFold(n_splits=cv1, shuffle=True)
                for n_comp in n_components_range:
                    val_scores = []
                    train_scores = []
                    for train_inner_idx, val_idx in inner_cv.split(X_rest):
                        X_train, X_val = X_rest[train_inner_idx], X_rest[val_idx]
                        y_train, y_val = y_rest[train_inner_idx], y_rest[val_idx]
                        
                        # Train model
                        model = PLSRegression(n_components=n_comp)
                        model.fit(X_train, y_train)
                        
                        # Calculate training performance
                        y_train_pred = model.predict(X_train)
                        train_metrics = calculate_metrics(y_train, y_train_pred)
                        train_scores.append(train_metrics)
                        
                        # Calculate validation performance
                        y_val_pred = model.predict(X_val)
                        val_metrics = calculate_metrics(y_val, y_val_pred)
                        val_scores.append(val_metrics)
                    
                    # Calculate average performance
                    mean_train_metrics = {k: np.mean([s[k] for s in train_scores]) for k in train_metrics}
                    mean_val_metrics = {k: np.mean([s[k] for s in val_scores]) for k in val_metrics}
                    
                    # Store inner CV performance
                    for metric in ['auc', 'acc', 'sen', 'spc']:
                        inner_cv_performances['train'][metric].append(mean_train_metrics[metric])
                        inner_cv_performances['val'][metric].append(mean_val_metrics[metric])
                    
                    # Use validation AUC to select best component number
                    mean_val_auc = mean_val_metrics['auc']
                    if mean_val_auc > best_score:
                        best_score = mean_val_auc
                        best_n_components = n_comp
                
                # Train final model with best parameters
                final_model = PLSRegression(n_components=best_n_components)
                final_model.fit(X_rest, y_rest)
                
                # Calculate VIP scores
                try:
                    vip_scores = calculate_vip(final_model, X_rest)
                    feature_importance.append(vip_scores)
                except Exception as e:
                    print(f"Warning: Error calculating VIP scores: {str(e)}")
                    feature_importance.append(None)
                
                # Predict test set
                y_test_pred = final_model.predict(X_test)
                test_predictions.extend(y_test_pred)
                test_true.extend(y_test)
                best_n_components_list.append(best_n_components)
        
        # Calculate average inner CV performance (using nanmean for NA values)
        avg_inner_cv_performance = {
            'train': {k: np.nanmean(v) if v else np.nan for k, v in inner_cv_performances['train'].items()},
            'val': {k: np.nanmean(v) if v else np.nan for k, v in inner_cv_performances['val'].items()}
        }
        
        return np.array(test_predictions), np.array(test_true), feature_importance, avg_inner_cv_performance
    except Exception as e:
        print(f"Error in double_cv: {str(e)}")
        return np.array([]), np.array([]), [], {'train': {}, 'val': {}}

def permutation_test(orig_metrics, X, y, n_components_range, n_permutations=1000, **kwargs):
    """
    Perform permutation test to evaluate model performance metrics.
    
    Args:
        orig_metrics (dict): Original performance metrics
        X (np.ndarray): Feature matrix
        y (np.ndarray): Target labels
        n_components_range (list): Range of components to test
        n_permutations (int): Number of permutations to perform
        **kwargs: Additional arguments for double_cv
        
    Returns:
        tuple: Null distribution metrics and p-values
    """
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
            n_components_range=n_components_range,
            **kwargs
        )
        
        # Calculate metrics only if predictions are not empty
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
    Prepare balanced dataset for a specific time group.
    
    Args:
        df (pd.DataFrame): Input dataframe
        time_group (str): Time group to prepare data for
        
    Returns:
        tuple: Balanced dataframe and labels
    """
    # Get disease group samples
    disease_samples = df[df['time_group'] == time_group]
    
    # Randomly sample same number of samples from healthy group
    healthy_samples = df[df['time_group'] == 'healthy']
    
    # Combine samples
    balanced_df = pd.concat([disease_samples, healthy_samples])
    
    # Prepare labels (0: healthy, 1: disease)
    y = (balanced_df['disease'] == 1).astype(int)
    
    return balanced_df, y

def main(input_file, output_dir, repeat=50, cv2=5, cv1=5, n_repeats_cv2=5, run_permutation=True):
    """
    Main function to run PLS-DA analysis.
    
    Args:
        input_file (str): Path to input CSV file
        output_dir (str): Path to output directory
        repeat (int): Number of downsample repeats
        cv2 (int): Number of outer CV folds
        cv1 (int): Number of inner CV folds
        n_repeats_cv2 (int): Number of repeats for outer CV
        run_permutation (bool): Whether to run permutation test
    """
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
    
    # Feature combinations
    feature_types = ['spc', 'spc+my', 'spc+scc', 'spc+parity', 'spc+my+scc+parity', 'spc_de', 'spc_rmR4']

    # Time groups
    time_groups = ['>10', '10-8', '7-6', '5-4', '3', '2', '1', '0']

    # Define component numbers to test
    n_components_range = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    
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
            
            # Store metrics for each repeat
            repeat_metrics = {
                'outer_auc': [], 'outer_acc': [], 'outer_sen': [], 'outer_spc': [],
                'inner_train_auc': [], 'inner_train_acc': [], 'inner_train_sen': [], 'inner_train_spc': [],
                'inner_val_auc': [], 'inner_val_acc': [], 'inner_val_sen': [], 'inner_val_spc': []
            }
            
            # Prepare complete dataset for permutation test
            balanced_data_full = balanced_df.copy()
            
            if feature_type == 'spc_de':
                spectral_type = 'derivative'
                base_feature = 'spc'
            elif feature_type == 'spc_rmR4':
                spectral_type = 'rmR4'
                base_feature = 'spc'
            else:
                spectral_type = 'original'
                base_feature = feature_type
            
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
                if feature_type == 'spc_de':
                    spectral_type = 'derivative'
                    base_feature = 'spc'
                elif feature_type == 'spc_rmR4':
                    spectral_type = 'rmR4'
                    base_feature = 'spc'
                else:
                    spectral_type = 'original'
                    base_feature = feature_type
                
                balanced_spectral = get_spectral_data(balanced_data, type=spectral_type)
                
                X = prepare_features(balanced_data, balanced_spectral, feature_type=base_feature)
                y = balanced_data['disease'].values
                
                # Perform double cross-validation
                pred, true, importance, inner_cv_perf = double_cv(
                    X, y, 
                    n_components_range=n_components_range,
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
                                    'vip': fold_importance[wave_idx]
                                })
            
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
                    n_components_range=n_components_range,
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
        importance_df = pd.DataFrame(columns=['time_group', 'feature', 'repeat', 'fold', 'wavenumber', 'vip'])
        
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
