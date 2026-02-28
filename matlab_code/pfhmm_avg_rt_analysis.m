function st = pfhmm_avg_rt_analysis(experiment, postfix, dynamic_mode)
   
    if nargin < 3
        experiment = "sealion"; % turtle, sealion
        postfix = "";
        dynamic_mode = 'median';
    end

     %% Set working directory and file paths
    currentFolder = fileparts(mfilename('fullpath'));
    cd(currentFolder); % Ensure relative paths resolve correctly

    fdir = fullfile('..', 'mat_data', sprintf('experiment_%s', experiment));
    output_fname  = fullfile(fdir, sprintf('%s_%s%s_%s.mat', mfilename, experiment, postfix, dynamic_mode)); % File to store regression results
    fprintf("Output file name: %s \n", output_fname);

    fname_pfhmm  = fullfile(fdir, sprintf('pfhmm_rho_fit_%s%s.mat', experiment, postfix));
    fprintf("Input file name: %s \n", fname_pfhmm);

    %% If output file does not exist, compute the necessary statistics
    if ~exist(output_fname, 'file')    
        % Load experimental data for the 'sealion' experiment using get_data.
        [data, ~] = get_data(experiment);
        
        % Check if the pfhmm data file exists. If not, compute it.
        if ~exist(fname_pfhmm, 'file')
            fprintf("Need to fit phfmm using pfhmm_fit.");
        end
        % Load the pfhmm time-series results.
        f = load(fname_pfhmm);
        if strcmp(dynamic_mode, 'median')
            dynamics = f.median_dynamics;
        else
            dynamics = f.mean_dynamics;
        end
        
        Nsub = numel(f.obs_dynamics);
        [T, B] = size(f.obs_dynamics{1}.vol);

        y_std_mean = dynamics.y_std;
        y_mean_mean = dynamics.y_mean;
        vol_mean = dynamics.vol;
        sto_mean = dynamics.sto;

        %% Prepare variables for regression analysis
        % Define time indices for regression:
        % t1: indices for predictors (all but the last time point)
        % t: indices for response variable (all but the first time point)
        t1 = 1:(T-1);
        t = 2:T;
        
        % Preallocate the regression coefficients for each subject.
        b_rt = nan(length(data), 6);
        % Define regressor names for the GLM (the first coefficient is the intercept).
        regressor_names = {'intercept', 'y_std', 'y_mean', 'vol', 'sto', 'accuracy'};
        
        %% Loop over subjects and fit a GLM to their response times
        for n = 1:Nsub
            % Response times
            rt = data{n}.response_time(t, :);
            rt = rt(:);
            rt(isoutlier(rt)) = NaN;
        
            % Accuracy (subject-specific)
            accuracy = (data{n}.choice == data{n}.outcome);
            acc = accuracy(t1, :);
            acc = acc(:);
        
            % Task-level regressors (same for all subjects)
            y_std  = y_std_mean(t1, :);   y_std  = y_std(:);
            y_mean = y_mean_mean(t1, :);  y_mean = y_mean(:);
            vol    = vol_mean(t1, :); vol   = vol(:);
            sto    = sto_mean(t1, :); sto   = sto(:);
        
            % Valid trials
            valid = ~isnan(rt);
        
            % Normalize RT
            y = rt(valid) / 10000;
        
            % Design matrix
            x = [ ...
                y_std(valid), ...
                y_mean(valid), ...
                vol(valid), ...
                sto(valid), ...
                acc(valid) ...
            ];
        
            % Fit GLM
            b_rt(n, :) = glmfit(x, y);
        end
        
        % Save the regression coefficients and regressor names to a .mat file.
        save(output_fname, 'b_rt', 'regressor_names');
    end

    %% Load the regression coefficients for further statistical analysis.
    f = load(output_fname);
    b_rt = f.b_rt;
    regressor_names = f.regressor_names;

    %% Run statistical tests on the regression coefficients
    % Perform a one-sample t-test on the regression coefficients across subjects.
    [~, pval_eff, ~, st] = ttest(b_rt);
    tval_eff = st.tstat;       % Extract t-values from the t-test results.
    mean_eff = mean(b_rt);     % Compute the mean of the regression coefficients.
    serr_eff = serr(b_rt);     % Compute the standard error of the mean (SER) for each coefficient.

    %% Organize statistical outputs into a table structure.
    tbl_data = [mean_eff; serr_eff; tval_eff; pval_eff];
    st.table.rows = {'Mean Effect', 'S.E.M.', 't-statistics', 'P-value'};
    st.table.columns = regressor_names;
    st.table.data = tbl_data;
    st.p      = pval_eff;
    st.mean   = mean_eff;
    
    T = array2table(st.table.data, 'VariableNames', st.table.columns, 'RowNames', st.table.rows);
    fprintf('\n=== Supplementary Table: Response Time Analysis Table ===\n');
    disp(T);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Helper Function: Compute the standard error (SER) of a matrix.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function se = serr(x)
    % SERR Computes the standard error of the mean across rows.
    se = std(x) ./ sqrt(size(x, 1));
end


