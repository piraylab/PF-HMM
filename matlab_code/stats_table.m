function stats_table
    
    addpath('tools');
    fdir = fullfile('..', 'mat_data');
    
    %% Supplementary Table 1 (HMM Param Recovery Correlation)
    st = hmm_recovery(false);
    
    copy_table(st.table.data, 2);
    
    % Create a table using the assembled data.
    T = array2table(st.table.data, 'RowNames', st.table.rows, 'VariableNames', st.table.columns);
    
    % Display the table of parameter recovery in the Command Window
    fprintf('\n=== Supplementary Table 1: HMM Parameter Recovery Correlation Table ===\n');
    disp(T);
    
    
    %% Supplementary Table 2: Fitted HMM Parameters Statistics
    % Load the fitted HMM parameters from an external file
    f = load(fullfile(fdir, "hmm_params.mat"));
    hmm_params = f.parameters;
    
    % Compute quantile statistics (25%, median, and 75%) for each column of hmm_params
    q25 = quantile(hmm_params, 0.25);
    q50 = quantile(hmm_params, 0.50);
    q75 = quantile(hmm_params, 0.75);
    statsMatrix = [q25; q50; q75];
    % Transpose it so that each row corresponds to a specific parameter.
    statsMatrix = statsMatrix';
    
    % Define row names: first 4 rows for Volatility (Block1 to Block4),
    % next 4 for Stochasticity, and 4 rows for Beta.
    rowNames = {...
        'Volatility_{Block1}', 'Volatility_{Block2}', 'Volatility_{Block3}', 'Volatility_{Block4}', ...
        'Stochasticity_{Block1}', 'Stochasticity_{Block2}', 'Stochasticity_{Block3}', 'Stochasticity_{Block4}'};
    % Define column names for the statistics.
    colNames = {'25% quantile','Median','75% quantile'};
    
    copy_table(statsMatrix, 2);
    
    % Create a table using the assembled data.
    T = array2table(statsMatrix, 'RowNames', rowNames, 'VariableNames', colNames);
    
    % Display the table of parameter recovery in the Command Window
    fprintf('\n=== Supplementary Table 2: Fitted HMM Parameters Statistics Table ===\n');
    disp(T);
    
    %% Supplementary Table 3: Fitted HMM Regression Effect
    fname = fullfile(fdir, 'hmm_params.mat'); 
    f = load(fname);
    lr = f.lr;
    block_effect = f.block_effect;
    st = compute_effect(lr, block_effect);
    
    copy_table(st.table.data, 2);
    
    T = array2table(st.table.data, 'VariableNames', st.table.columns, ...
            'RowNames', st.table.rows);
    % Display the table of parameter recovery in the Command Window
    fprintf('\n=== Supplementary Table 3: Fitted HMM Regression Effect Table ===\n');
    disp(T);
    
    %% Supplementary Table 4: Response Time Analysis
    st = response_time_analysis;
    copy_table(st.table.data, 2);
    
    T = array2table(st.table.data, 'VariableNames', st.table.columns, 'RowNames', st.table.rows);
    fprintf('\n=== Supplementary Table 4: Response Time Analysis Table ===\n');
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

