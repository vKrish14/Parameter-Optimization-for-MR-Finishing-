# Parameter-Optimization-for-MR-Finishing-
A MATLAB workflow for optimizing Magnetorheological (MR) finishing process parameters using Artificial Neural Networks (ANNs), Principal Component Analysis (PCA), and robust data cleaning. This code achieves accurate prediction and optimization of surface roughness improvements, with comprehensive diagnostics and reproducibility.
Overview

This repository provides a full pipeline to:

    Clean and preprocess experimental MR finishing data

    Engineer features and apply dimensionality reduction (PCA)

    Train a multi-layer ANN to predict surface roughness improvement

    Optimize process parameters via numerical optimization (fminsearch)

    Generate diagnostic plots and export results for transparency

Features

    Data Cleaning: Removes missing values, duplicate rows, and statistical outliers (IQR-based) for dataset integrity.

    Feature Engineering: Expands features with polynomial and interaction terms, reducing collinearity and increasing predictive power.

    Dimensionality Reduction: PCA retains the most informative input components, minimizing overfitting.

    ANN Architecture: Feedforward ANN with three hidden layers ([30,rained by scaled conjugate gradient (SCG) and early stopping for best correlation.

    Optimization: Predicts and identifies optimal process parameters by maximizing model output.

    Comprehensive Diagnostics: Generates line and scatter plots, absolute error, residual analysis, and model learning curves.

    Results Export: All predictions and metrics saved to Excel for reproducibility.

Requirements

    MATLAB R2018b or newer (recommended for Neural Network Toolbox compatibility)

    Statistics and Machine Learning Toolbox

Usage

    Prepare Data

        Ensure your experimental input data is in input.xlsx (columns: T, W, S, normalized to ).

    Place your corresponding output data in output.xlsx (column: Percentage change).

Run the Script

    matlab
    % Main script
    MR_Finishing_ANN_Optimization.m

    The script will:

        Load and clean the provided .xlsx files

        Engineer features and apply PCA

        Train the ANN with multiple restarts for optimal R and MSE

        Optimize process settings for maximal predicted improvement

        Generate diagnostic plots

        Save results to ANN_fminsearch_Cleaned_IQR_PCA_Results.xlsx

    Interpret Results

        Optimized parameters and maximum predicted output are displayed in the MATLAB console.

        Diagnostic plots illustrate model accuracy and prediction confidence.

        Full results (actual, predicted, errors) are available in the exported Excel file.

Plot Outputs

The code automatically generates the following visuals for model interpretation:

    Actual vs. Predicted Output

    Absolute Error per Sample

    Histogram of Residuals

    Scatter: Actual vs. Predicted

    Residuals vs. Predicted

    Distribution of Actual Values

    Output vs. Target

    Mean Squared Error vs. Epoch

Example Output

text
=== OPTIMAL INPUTS FOR MAXIMUM "Percentage change" after Cleaning, IQR, PCA ===
Best T (Tool rotation):      0.745
Best W (Workpiece rotation): 0.255
Best S (Feed rate):          0.800
Predicted Maximum Output:    0.993
========================================================
Results saved to ANN_fminsearch_Cleaned_IQR_PCA_Results.xlsx

Customization

    Update ANN architecture by editing the [30 20 10] vector in the code.

    Adjust cleaning rules or PCA variance threshold as needed for your dataset.

    Edit plotting sections to customize visualization style or outputs.

Citation

If you use this code for research or publication, please cite the corresponding article or acknowledge the methodology as:

    "Process parameter optimization for MR finishing using ANN surrogate modeling, PCA, and robust data cleaning, with full transparency and diagnostics."

License

This project is provided for academic and research use. For other uses, please request permission from the author.

Contact: For questions or support, please open an issue or contact the repository maintainer.
