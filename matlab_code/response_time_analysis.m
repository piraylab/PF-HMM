function st = response_time_analysis
% RESPONSE_TIME_ANALYSIS Analyzes response times by fitting a GLM per subject.
%
%   stats = response_time_analysis
%
%   This function performs the following:
%     1. Loads experimental data using get_data.
%     2. Checks if pre-computed particle filter results ("pfhmm_sealion_timeseries")
%        exist; if not, it computes time-series from the particle filter (pfhmm) for simulated data.
%     3. Uses these pfhmm time-series (volatility, stochasticity, and their standard 
%        deviations) as regressors to fit a GLM on the subjects’ response times.
%     4. Computes statistical measures (mean, standard error, t-values, p-values) 
%        for the regression coefficients.
%     5. Organizes the results into a table and then plots one of the regression coefficients.
%
%   The function saves and loads intermediate results into/from .mat files for
%   efficiency and reproducibility.

    %% Set working directory and file paths
    currentFolder = fileparts(mfilename('fullpath'));
    cd(currentFolder); % Ensure relative paths resolve correctly
    fdir = fullfile('..', 'mat_data'); % Directory containing .mat files
    fname  = fullfile(fdir, sprintf('%s.mat', mfilename)); % File to store regression results

    %% If output file does not exist, compute the necessary statistics
    if ~exist(fname, 'file')    
        % Load experimental data for the 'sealion' experiment using get_data.
        [data, ~] = get_data('sealion');
        
        % Check if the pfhmm data file exists. If not, compute it.
        fname_pfhmm  = fullfile(fdir, sprintf('pfhmm_lr.mat'));
        if ~exist(fname_pfhmm, 'file')
            % Use the outcome data from the first subject (can be adapted as needed)
            outcome = data{1}.outcome;
            % Define true metadata for volatility and stochasticity.
            metadata = struct('true_sto', [0.1250 0.1250 0.2500 0.2500], ...
                              'true_vol', [0.0250 0.1000 0.0250 0.1000]);
            num_sim = 100; % Number of simulations to average over
            % Initialize a structure for accumulating pfhmm variables.
            vars = struct('vol', 0, 'sto', 0, 'val', 0, 'y_std', 0, 'y_mean', 0, 'learning_rate', 0);
            % Preallocate the matrix for learning rates from pfhmm results
            lr_pfhmm = nan(num_sim, 4);
            
            % Loop over simulations to compute particle filter estimates
            for i = 1:num_sim            
                rng(i); % Set seed for reproducibility
                % Set pfhmm parameters for both stochasticity and volatility noise.
                params.sigma_sto = 0.02;
                params.sigma_vol = 0.02;
                params.s0 = mean(metadata.true_sto);
                params.v0 = mean(metadata.true_vol);
                % Run the particle filter HMM on the outcome data
                [~, vars_i] = pfhmm(params, outcome);
                % Record the estimated learning rate (per block)
                lr_pfhmm(i, :) = vars_i.learning_rate;
                % Average the time-series variables over simulations
                snames = fieldnames(vars_i);
                for j = 1:length(snames)
                    % Accumulate each field of vars_i, averaging over num_sim simulations.
                    vars.(snames{j}) = vars.(snames{j}) + vars_i.(snames{j})/num_sim;
                end        
            end
            % Save the computed pfhmm time-series, learning rates, and outcome data for re-use.
            save(fname_pfhmm, 'vars', 'lr_pfhmm', 'outcome');
        end
        % Load the pfhmm time-series results.
        f = load(fname_pfhmm);
        outcome = f.outcome;    
        vars = f.vars;

        %% Prepare variables for regression analysis
        % Calculate the number of time steps from the outcome data.
        stats = size(outcome, 1);
        % Define time indices for regression:
        % t1: indices for predictors (all but the last time point)
        % t: indices for response variable (all but the first time point)
        t1 = 1:(stats-1);
        t = 2:stats;
        % Extract variables for regression:
        % y_std, y_mean: variability measures from pfhmm output, for time steps t1.
        y_std = vars.y_std(t1, :);
        y_mean = vars.y_mean(t1, :);
        % vol and sto: estimates of volatility and stochasticity, excluding the last time point.
        vol = vars.vol(1:end-1, :);
        sto = vars.sto(1:end-1, :);
        
        % Convert matrices to column vectors for use as regressors.
        y_std = y_std(:);
        y_mean = y_mean(:);
        vol = vol(:);
        sto = sto(:);
        
        % Preallocate the regression coefficients for each subject.
        b_rt = nan(length(data), 6);
        % Define regressor names for the GLM (the first coefficient is the intercept).
        regressor_names = {'intercept', 'y_std', 'y_mean', 'vol', 'sto', 'accuracy'};
        
        %% Loop over subjects and fit a GLM to their response times
        for n = 1:length(data)
            % Extract response times for subject n at time indices t.
            rt = data{n}.response_time(t, :);    
            rt = rt(:); % Convert to a column vector
            
            % Compute accuracy: a binary indicator where choice equals outcome.
            accuracy = (data{n}.choice) == data{n}.outcome;
            % Use predictors corresponding to time indices t1.
            accuracy = accuracy(t1, :);
            accuracy = accuracy(:); % Vectorize accuracy
            
            % Remove NaN values in response times for valid regression.
            nans = isnan(rt);
            % Normalize response time (dividing by 10000 as per data scaling).
            y = rt(~nans) / 10000;
            % Construct a design matrix from the pfhmm outputs and accuracy.
            x = [y_std(~nans) y_mean(~nans) vol(~nans) sto(~nans) accuracy(~nans)];
            % Fit a generalized linear model (GLM) to predict normalized response time.
            b_rt(n, :) = glmfit(x, y);
        end
        
        % Save the regression coefficients and regressor names to a .mat file.
        save(fname, 'b_rt', 'regressor_names');
    end

    %% Load the regression coefficients for further statistical analysis.
    f = load(fname);
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
    st.labels = {'PE','PE x Sto', 'PE x Vol', 'PE x Sto x Vol', ...
                 'Sto', 'Vol', 'Sto x Vol', 'Intercept'};
    st.mean   = mean_eff;
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Helper Function: Compute the standard error (SER) of a matrix.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function se = serr(x)
    % SERR Computes the standard error of the mean across rows.
    se = std(x) ./ sqrt(size(x, 1));
end


