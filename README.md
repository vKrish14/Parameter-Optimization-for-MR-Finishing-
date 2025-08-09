# Parameter-Optimization-for-MR-Finishing-
A MATLAB workflow for optimizing Magnetorheological (MR) finishing process parameters using Artificial Neural Networks (ANN), Principal Component Analysis (PCA), and robust data cleaning. This pipeline enables accurate prediction and optimization of surface roughness improvement with full diagnostics and transparency.
Features

    Data Cleaning: Removes missing values, duplicates, and outliers (IQR-based).

    Feature Engineering: Polynomial and interaction terms expand input power.

    Dimensionality Reduction: PCA retains ≥95% variance.

    ANN: Three hidden layers ([30 20 10]), trained via scaled conjugate gradient with early stopping.

    Optimization: fminsearch finds parameters maximizing predicted output.

    Diagnostics: Generates plots (actual vs. predicted, error, residuals, MSE vs. epoch).

    Reproducibility: Results exported to Excel.

Requirements

    MATLAB R2018b or newer

    Statistics and Machine Learning Toolbox

Usage

    Input Files:
    input.xlsx — columns: T, W, S (normalized, no header change)
    output.xlsx — column: Percentage change

    Run the Script:
    Open and run the MATLAB script in this repo.

    View Output:

        Optimized parameters and output visible in MATLAB console

        Diagnostic plots generated automatically

        Results exported to ANN_fminsearch_Cleaned_IQR_PCA_Results.xlsx

Plot Outputs

    Actual vs. Predicted Output

    Absolute Error per Sample

    Residuals (histogram and scatter)

    Output vs. Target

    Mean Squared Error vs. Epoch

Example Output

text
Best T: 0.745

Best W: 0.255

Best S: 0.800
Predicted Maximum Output: 0.993
Results saved to ANN_fminsearch_Cleaned_IQR_PCA_Results.xlsx

Citation

If you use this code, please cite as:

    "Parameter Optimization of Magnetorheological Finishing Using Machine Learning for Curved Surface "


Academic/research use only.
For other use, please contact the author.
