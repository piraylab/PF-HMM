function stats_table
    
    addpath('tools');  
    fdir = fullfile('..', 'mat_data');

    %% ----- Main Tables: Model Comparisons -----
    fprintf('\n=== Table: Model Comparison Table (sealion) ===\n');
    st = other_bmc('sealion', '');
    copy_table(st.table_struct.data, 3);

    fprintf('\n=== Table: Model Comparison Table (turtle) ===\n');
    st = other_bmc('turtle', '');
    copy_table(st.table_struct.data, 3);

    % Jang et al., 2019
    fprintf('\n=== Table: Model Comparison Table (binB) ===\n');
    st = other_bmc('binB', '');
    copy_table(st.table_struct.data, 3);

    % Piray et al., 2019
    fprintf('\n=== Table: Model Comparison Table (binA) ===\n'); 
    st = other_bmc('binA', '');
    copy_table(st.table_struct.data, 3);
    
    %% ---------- Supplementary Table 1 (HMM Param Recovery) ----------
    fprintf('\n=== Supplementary Table 1: HMM Parameter Recovery Table ===\n');
    st = hmm_rho_recovery(false);
    copy_table(st.table.data, 3);
    % Create a table using the assembled data.
    T = array2table(st.table.data, 'RowNames', st.table.rows, 'VariableNames', st.table.columns);
    % Display the table of parameter recovery in the Command Window
    disp(T);

    %% ---------- Supplementary Table 2: Fitted HMM (Rho) Parameters Statistics ----------
    fprintf('\n=== Supplementary Table 2: Fitted HMM (Rho) Parameters Statistics (sea lion) ===\n');
    % Load the fitted HMM parameters from an external file
    f = load(fullfile(fdir, 'experiment_sealion', "hmm_rho_params.mat"));
    hmm_params = f.parameters;
    statsMatrix = [mean(hmm_params); std(hmm_params)];
    % Transpose it so that each row corresponds to a specific parameter.
    statsMatrix = statsMatrix';
    % Define row names: first 4 rows for Volatility (Block1 to Block4), next 4 for Stochasticity, and Rho.
    rowNames = {'Volatility_{Block1}', 'Volatility_{Block2}', 'Volatility_{Block3}', 'Volatility_{Block4}', ...
        'Stochasticity_{Block1}', 'Stochasticity_{Block2}', 'Stochasticity_{Block3}', 'Stochasticity_{Block4}', ...
        'Rho'};
    % Define column names for the statistics.
    colNames = {'Mean', 'STD'};
    copy_table(statsMatrix, 3);
    % Create a table using the assembled data.
    T = array2table(statsMatrix, 'RowNames', rowNames, 'VariableNames', colNames);
    % Display the table of parameter recovery in the Command Window
    disp(T);

    fprintf('\n=== Supplementary Table 2: Fitted HMM (Rho) Parameters Statistics (turtle) ===\n');
    % Load the fitted HMM parameters from an external file
    f = load(fullfile(fdir, 'experiment_turtle', "hmm_rho_params.mat"));
    hmm_params = f.parameters;
    statsMatrix = [mean(hmm_params); std(hmm_params)];
    statsMatrix = statsMatrix';
    rowNames = {'Volatility_{Block1}', 'Volatility_{Block2}', 'Volatility_{Block3}', 'Volatility_{Block4}', ...
        'Stochasticity_{Block1}', 'Stochasticity_{Block2}', 'Stochasticity_{Block3}', 'Stochasticity_{Block4}', ...
        'Rho'};
    colNames = {'Mean', 'STD'};
    copy_table(statsMatrix, 3);
    T = array2table(statsMatrix, 'RowNames', rowNames, 'VariableNames', colNames);
    disp(T);
   
    %% ---------- Supplementary Table 3: Fitted HMM (Rho) Regression Effect (sea lion)
    fprintf('\n=== Supplementary Table 3: Fitted HMM (Rho) Regression Effect (sea lion) ===\n');
    fname = fullfile(fdir, 'experiment_sealion', 'hmm_rho_params.mat'); 
    f = load(fname);
    lr = f.lr;
    block_effect = f.block_effect;
    st = compute_effect(lr, block_effect);
    copy_table(st.table.data, 3);
    T = array2table(st.table.data, 'VariableNames', st.table.columns, ...
            'RowNames', st.table.rows);
    % Display the table of parameter recovery in the Command Window
    disp(T);

    %% ---------- Supplementary Table 4: PF-HMM (Rho) Param Recovery ----------
    fprintf('\n=== Supplementary Table 4: PF-HMM Parameter Recovery ===\n');
    st = pfhmm_rho_recovery(false);
    copy_table(st.table.data, 3);
    % Create a table using the assembled data.
    T = array2table(st.table.data, 'RowNames', st.table.rows, 'VariableNames', st.table.columns);
    % Display the table of parameter recovery in the Command Window
    disp(T);

    %% ---------- Supplementary Table 5: PF-HMM (Rho) Response Time Analysis (sea lion) ----------
    fprintf('\n=== Supplementary Table 5: PF-HMM Response Time Analysis (sea lion) ===\n');
    st = pfhmm_avg_rt_analysis('sealion', '', 'median');
    copy_table(st.table.data, 3);

    %% ---------- Supplementary Table 6: Fitted HMM (Rho) Regression Effect (turtle)
    fprintf('\n=== Supplementary Table 6: Fitted HMM (Rho) Regression Effect (turtle) ===\n');
    fname = fullfile(fdir, 'experiment_turtle', 'hmm_rho_params.mat'); 
    f = load(fname);
    lr = f.lr;
    block_effect = f.block_effect;
    st = compute_effect(lr, block_effect);
    copy_table(st.table.data, 3);
    T = array2table(st.table.data, 'VariableNames', st.table.columns, ...
            'RowNames', st.table.rows);
    % Display the table of parameter recovery in the Command Window
    disp(T);
    
    %% ---------- Supplementary Table 7: PF-HMM (Rho) Response Time Analysis (turtle) ----------
    fprintf('\n=== Supplementary Table 7: PF-HMM Response Time Analysis (turtle) ===\n');
    st = pfhmm_avg_rt_analysis('turtle', '', 'median');
    copy_table(st.table.data, 3);

    %% ---------- Supplementary Table 8: Model Comparison (pfhmm vs. pfhmm+rho) ----------
    fprintf('\n=== Supplementary Table 8: Model Comparison (pfhmm vs. pfhmm+rho) ===\n');
    st = pfhmm_rho_bmc();
    copy_table(st.table_struct.data, 3);

    %% ---------- Supplementary Table 9: Fitted PF-HMM (Rho) Parameters Statistics ----------
    fprintf('\n=== Supplementary Table 9: Fitted PF-HMM (Rho) Parameters Statistics ===\n');
    st = pfhmm_rho_fit('sealion', '');
    copy_table(st.st_params.data, 3);
    T = array2table(st.st_params.data, 'VariableNames', st.st_params.columns, 'RowNames', st.st_params.rows);
    disp(T);
    st = pfhmm_rho_fit('turtle', '');
    copy_table(st.st_params.data, 3);
    T = array2table(st.st_params.data, 'VariableNames', st.st_params.columns, 'RowNames', st.st_params.rows);
    disp(T);

end

function str = copy_table(x, n)
    % This function rounds each row of matrix x to n decimal places.
    % If n is a single number, it is used for every row.
    % The rounded matrix is then converted to a string using num2clip.

    % If n is a scalar, create a vector with the same value for every row.
    if length(n) == 1
        n = n * ones(size(x, 1), 1);
    end

    % Round each row of x to the corresponding decimal places in n.
    for i = 1:size(x, 1)
        y(i, :) = round(x(i, :) * 10^n(i)) / 10^n(i);
    end

    % Copy the matrix to clipboard. 
    str = num2clip(y);
end

