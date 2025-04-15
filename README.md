# Milk-FTIR-Spectra-Subsequent-Disease-Prediction-Project

This project investigates the feasibility of using milk Fourier-transform infrared (FTIR) spectroscopy and machine learning to predict early-lactation diseases in Holstein dairy cows. To ensure unbiased evaluation, we adopted a repeated down-sampled double cross-validation framework that balances class distributions and integrates both model selection and assessment through nested 5-fold cross-validation. 

<img src="https://github.com/lindan1128/Milk-FTIR-Spectra-Health-Prediction-Project/blob/main/workflow.png" alt="Workflow diagram">

### Project structure
	Main/
	├── Codes/                        # Folder for codes for analysis and visualization
	│   ├── Figure 1C-1F.r            # codes for generating results for Figure 1C-1F
	│   ├── Figure 2.r                # codes for generating results for Figure 2A-2B
	│   ├── Figure 3A-3C.r            # codes for generating results for Figure 3A-3C
	│   ├── Figure 3D-3E.r            # codes for generating results for Figure 3D-3E
	├── Supplemental_Table/           # Folder for supplemental tables
	│   ├── Supplemental Table 1      
	│   ├── Supplemental Table 2
	│   ├── Supplemental Table 3
	│   └── Supplemental Table 4
	├── Supplemental_Figure/          # Folder for supplemental figures
	│   ├── Supplemental Figure 1      
	│   └── Supplemental Figure 2
	│── pls-da.py                     # Main function for PLS-DA
	│── ridge.py                      # Main function for ridge regression
	│── rf.py                         # Main function for random forest
	│── lstm.py                       # Main function for LSTM
	├── README.md                     # Readme file
	└── requirements.txt              # Dependencies
	
### Pseudocode for modeling -- PLS-DA as example

	Input Parameters:
    input_file: Path to input CSV file
    output_dir: Path to output directory
    repeat: Number of downsample repeats (default=50)
    cv2: Number of outer CV folds (default=5)
    cv1: Number of inner CV folds (default=5)
    n_repeats_cv2: Number of repeats for outer CV (default=5)
    run_permutation: Whether to run permutation test (default=True)

	Initialization:
    Create output directory
    Read CSV file into dataframe df
    Call process_data(df) to process data:
        - Check for NA values
        - Reclassify parity
        - Keep only samples with disease 0 and 1
        - Create time periods for disease group
        - Add time group column
        - Convert time group to numeric format
    
    Initialize result storage lists:
        all_results: Performance metrics
        all_importance: Feature importance
        all_permutations: Permutation test results
        all_p_values: p-values
        skipped_cases: Skipped cases
    
    Define feature types: ['spc', 'spc+my', 'spc+scc', 'spc+parity', 'spc+my+scc+parity', 'spc_de', 'spc_rmR4']
    Define time groups: ['>10', '10-8', '7-6', '5-4', '3', '2', '1', '0']
    Define component range: [2, 3, 4, 5, 6, 7, 8, 9, 10]

	Main Loop:
    For each time group:
        Prepare dataset
        If sample count < 10:
            Record skipped case and continue to next time group
        
        For each feature type:
            Initialize repeat metrics storage
            
            For each repeat i (from 1 to repeat):
                Downsample data to balance classes
                Prepare features and labels
                Execute double cross-validation double_cv:
                    For each outer CV repeat (from 1 to n_repeats_cv2):
                        Outer CV: Split into training and test sets
                        Inner CV: For each component number:
                            - For each fold:
                                - Train model
                                - Calculate training and validation performance
                        - Select best component number
                    - Train final model with best component number
                    - Calculate VIP scores
                    - Predict test set
                Calculate performance metrics
                Store repeat results
            
            Calculate mean and confidence interval for each metric
            Store results
            
            If run_permutation is True:
                Execute permutation test
                Store permutation results and p-values

	Save Results:
    Convert results to DataFrames and save to CSV files:
        pre.csv: Performance results
        imp.csv: Feature importance
        permu.csv: Permutation test results
        p_values.csv: p-values
        skipped.csv: Skipped cases

	Print Statistics:
    Number of time groups
    Number of features
    Samples per time group
    Number of importance scores
    Number of wavelengths
    Number of permutation tests
	
### Modeling
	
	## PLS-DA
	python pls-da.py --input_file INPUT_FILE --output_dir OUTPUT_DIR [options]
	## ridge regression
	python ridge.py --input_file INPUT_FILE --output_dir OUTPUT_DIR [options]
	## random forest
	python rf.py --input_file INPUT_FILE --output_dir OUTPUT_DIR [options]
	## LSTM
	python lstm.py --input_file INPUT_FILE --output_dir OUTPUT_DIR [options]
	
	The key hyperparameters for the model are:
	--input_file INPUT_FILE Path to the input CSV file containing spectral data
	--output_dir OUTPUT_DIR Directory to save output files
	--repeat REPEAT Number of downsample repeats (default: 50)
	--cv2 CV2 Number of outer CV folds (default: 5)
	--cv1 CV1 Number of inner CV folds (default: 5)
	--n_repeats_cv2 N_REPEATS Number of repeats for outer CV (default: 5)
	--run_permutation Flag to run permutation test

	The output for the model are:
	* pre.csv: Performance metrics for each time group and feature type
	* imp.csv: Feature importance scores
	* permu.csv: Permutation test results (if run_permutation is enabled)
	* p_values.csv: P-values from permutation tests (if run_permutation is enabled)
	* skipped.csv: Information about skipped cases due to insufficient samples

