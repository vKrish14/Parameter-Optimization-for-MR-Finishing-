%% MR Finishing Process Optimization using ANN + fminsearch with Full Data Cleaning on Normalized Data

clc; clear;

% Step 1: Load Data (variable names preserved)
input_tbl = readtable('input.xlsx', 'VariableNamingRule','preserve');
output_tbl = readtable('output.xlsx', 'VariableNamingRule','preserve');

% Extract input features and target (already normalized)
T = input_tbl.T; % Nx1
W = input_tbl.W;
S = input_tbl.S;
targets = output_tbl.("Percentage change");

data_all = [T, W, S, targets];

% --- Data Cleaning ---
% 1. Remove rows with any NaN values (if any)
nan_rows = any(isnan(data_all), 2);
if any(nan_rows)
    fprintf('Removing %d rows with NaNs.\n', sum(nan_rows));
    data_all(nan_rows, :) = [];
end

% 2. Remove duplicate rows
[unique_data, ~, ~] = unique(data_all, 'rows', 'stable');
if size(unique_data, 1) < size(data_all, 1)
    fprintf('Removed %d duplicate rows.\n', size(data_all, 1) - size(unique_data, 1));
end
data_all = unique_data;

% 3. Outlier Removal using IQR method (per column)
Q1 = quantile(data_all, 0.25);
Q3 = quantile(data_all, 0.75);
IQR = Q3 - Q1;
lower_bound = Q1 - 1.5*IQR;
upper_bound = Q3 + 1.5*IQR;
outlier_rows = any((data_all < lower_bound) | (data_all > upper_bound), 2);
if any(outlier_rows)
    fprintf('Removing %d outlier rows based on IQR.\n', sum(outlier_rows));
    data_all(outlier_rows, :) = [];
end

% Extract cleaned inputs and targets
T_clean = data_all(:, 1);
W_clean = data_all(:, 2);
S_clean = data_all(:, 3);
targets_clean = data_all(:, 4);

% --- Feature Engineering ---
inputs_orig = [T_clean, W_clean, S_clean]; % Nx3
TW = T_clean .* W_clean;
TS = T_clean .* S_clean;
WS = W_clean .* S_clean;
T2 = T_clean .^ 2;
W2 = W_clean .^ 2;
S2 = S_clean .^ 2;
inputs_aug = [inputs_orig, TW, TS, WS, T2, W2, S2]; % Nx9

% --- PCA on normalized augmented features ---
inputs_norm = normalize(inputs_aug);
[coeff, score, ~, ~, explained] = pca(inputs_norm);
cumExplained = cumsum(explained);
numPCs = find(cumExplained >= 95, 1, 'first');
fprintf('Selected %d principal components (%.2f%% variance explained)\n', numPCs, cumExplained(numPCs));
inputs_pca = score(:, 1:numPCs)';  % features x samples

% Normalize target (again, safeguard)
[targetn, targetSettings] = mapminmax(targets_clean', 0, 1);
% Normalize PCA components for ANN training
[inputn, inputSettings] = mapminmax(inputs_pca, 0, 1);

% --- Train ANN with early stop when R >= 0.95 ---
bestR = -Inf;
bestPerf = Inf;
retries = 2000;   % Allow max retries 
thresholdR = 0.96;
stop_training = false;

for i = 1:retries
    net = fitnet([30 20 10]);
    net.trainFcn = 'trainscg';
    net.trainParam.epochs = 1000;
    net.divideParam.trainRatio = 0.8;
    net.divideParam.valRatio = 0.1;
    net.divideParam.testRatio = 0.1;
    [net, tr] = train(net, inputn, targetn);
    predictedn = net(inputn);
    predicted_actual = mapminmax('reverse', predictedn, targetSettings);
    R = corrcoef(predicted_actual, targets_clean');
    perf = perform(net, targetn, predictedn);
    currR = R(1,2);
    if (currR > bestR) || (currR == bestR && perf < bestPerf)
        bestR = currR;
        bestNet = net;
        bestPerf = perf;
        bestPredicted = predicted_actual;
        bestTr = tr; % Store tr for the best run for MSE-vs-epoch plot
    end
    fprintf('Attempt %d: R = %.4f, MSE = %.6f\n', i, currR, perf);
    if currR >= thresholdR
        fprintf('Target R >= %.2f reached. Stopping training.\n', thresholdR);
        stop_training = true;
        break;
    end
end
net = bestNet;
predicted_actual = bestPredicted;

% --- fminsearch Optimization on PCA feature space ---
lb = zeros(numPCs, 1);
ub = ones(numPCs, 1);
objfun = @(x) -net(x(:)); % maximize output = minimize negative
x0 = 0.5 * ones(numPCs, 1);
fun = @(x) objfun(min(max(x, lb), ub));
opts = optimset('Display', 'off');
[best_input_norm, fval] = fminsearch(fun, x0, opts);
best_input_norm = min(max(best_input_norm, lb), ub);

% Reverse normalization in PCA space
best_input_pca = mapminmax('reverse', best_input_norm, inputSettings);
best_input_features_norm = coeff(:, 1:numPCs) * best_input_pca; % 9 x 1 vector
best_input_features_norm = best_input_features_norm';
mean_feats = mean(inputs_aug);
std_feats = std(inputs_aug);
best_input_features_denorm = best_input_features_norm .* std_feats + mean_feats;
best_T = best_input_features_denorm(1);
best_W = best_input_features_denorm(2);
best_S = best_input_features_denorm(3);
best_output = mapminmax('reverse', -fval, targetSettings);

% --- Display optimized inputs and predicted output ---
fprintf('\n=== OPTIMAL INPUTS FOR MAXIMUM "Percentage change" after Cleaning, IQR, PCA ===\n');
fprintf('Best T (Tool rotation):      %.6f\n', best_T);
fprintf('Best W (Workpiece rotation): %.6f\n', best_W);
fprintf('Best S (Feed rate):          %.6f\n', best_S);
fprintf('Predicted Maximum Output:    %.6f\n', best_output);
fprintf('========================================================\n');

% --- Plot diagnostics ---
N = length(targets_clean);
sample_idx = 1:N;
abs_error = abs(predicted_actual - targets_clean');
residuals = predicted_actual - targets_clean';

figure;
subplot(2,3,1);
plot(sample_idx, targets_clean, 'ko-', 'LineWidth', 1.5, 'MarkerFaceColor', 'k'); hold on;
plot(sample_idx, predicted_actual, 'r*-', 'LineWidth', 1.5);
legend('Actual', 'Predicted');
xlabel('Sample Index'); ylabel('Percentage Change');
title('Actual vs Predicted');
grid on;

subplot(2,3,2);
bar(sample_idx, abs_error, 'cyan');
xlabel('Sample Index'); ylabel('Absolute Error');
title('Absolute Error per Sample'); grid on;

subplot(2,3,3);
histogram(residuals, 8, 'FaceColor', [1 0.5 0]);
xlabel('Residual Value'); ylabel('Frequency');
title('Histogram of Residuals'); grid on;

subplot(2,3,4);
scatter(targets_clean, predicted_actual, 90, 'b', 'filled'); hold on;
plot([min(targets_clean) max(targets_clean)], [min(targets_clean) max(targets_clean)], 'r--', 'LineWidth', 1.5);
xlabel('Actual'); ylabel('Predicted');
title('Scatter: Actual vs Predicted'); grid on;

subplot(2,3,5);
scatter(predicted_actual, residuals, 70, 'g', 'filled'); hold on;
yline(0,'r--','LineWidth',1);
xlabel('Predicted'); ylabel('Residuals');
title('Residuals vs Predicted'); grid on;

subplot(2,3,6);
histogram(targets_clean, 8, 'FaceColor', [.5 .5 .5]);
xlabel('Actual Percentage Change'); ylabel('Frequency');
title('Distribution of Actual Values'); grid on;

sgtitle('ANN Regression Diagnostics with Data Cleaning, IQR, and PCA');

% --- ADDITIONAL REQUESTED PLOTS ---

%% Output vs Target Plot
figure;
plot(sample_idx, targets_clean, 'ko-', 'LineWidth', 1.5, 'MarkerFaceColor', 'k'); hold on;
plot(sample_idx, predicted_actual, 'r*-', 'LineWidth', 1.5);
legend('Actual (Target)','Predicted (Output)');
xlabel('Sample Index');
ylabel('Percentage Change');
title('Output vs Target');
grid on;

%% Mean Squared Error (MSE) vs Epochs Plot
if exist('bestTr','var') && isfield(bestTr,'perf')
    figure;
    plot(bestTr.perf, 'b-', 'LineWidth', 1.5);
    xlabel('Epoch');
    ylabel('Mean Squared Error (MSE)');
    title('Mean Squared Error vs Epochs');
    grid on;
end

% --- Export results ---
result_table = table(sample_idx', targets_clean, predicted_actual', abs_error', ...
    'VariableNames', {'Sample', 'Actual_PercentageChange', 'Predicted_PercentageChange', 'AbsoluteError'});
writetable(result_table, 'ANN_fminsearch_Cleaned_IQR_PCA_Results.xlsx');
fprintf('Results saved to ANN_fminsearch_Cleaned_IQR_PCA_Results.xlsx\n');
