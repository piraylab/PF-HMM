function tbl = binary_hgf_fit(experiment, postfix)
    if nargin < 1
        experiment = 'sealion'; 
        % 'sealion', 'turtle', 'binA', 'binB'
        postfix = ''; 
    end

    fdir = fullfile('..', 'mat_data', sprintf('experiment_%s', experiment));
    if ~exist(fdir,'dir'), error("Experiment folder not found: %s", fdir); end
    
    if strcmpi(experiment, 'sim')
        f = load(fullfile(fdir, sprintf("data_sim_binary_hgf%s.mat", postfix)));
        data = f.sim_data;
    elseif strcmpi(experiment, 'binA') || strcmpi(experiment, 'binB')
        data = get_data_bin(experiment);
    else
        [data] = get_data(experiment);
    end

    fname = fullfile(fdir, sprintf('%s_%s%s.mat', mfilename, experiment, postfix));

    N = numel(data);
    % blocks = size(data{1}.outcome, 2);

    if ~exist(fname, 'file')
        % nu must be >0; if ~1 (overfitting), kappa also ~1
        % large nu -> volatility beliefs fluctuate wildly
        % small nu -> volatility is stable
        % kappa >=0; if too large learning hypersensitive
        % kappa = 0 -> volatility irrelevant
        % large kappa -> volatility dominates learning 
        % omega more neg -> stable environment
        % omega less neg or pos -> volatile environment
        if strcmp(experiment, 'binA')
            % Bias-based response model
            % parameters: [nu, kappa, omega, bv, be, bi]
            lb = [1e-4  0   -5  -10  -10  -10];
            ub = [2   2   +5   10   10   10];
            param_names = {'nu', 'kappa', 'omega','bv','be','bi'};
        elseif strcmp(experiment, 'binB') 
            % Beta-based response model
            % 4 parameters: [nu, kappa, omega, temp]
            lb = [1e-4  0   -5  -5];
            ub = [2   2   +5     5];
            param_names = {'nu', 'kappa', 'omega','temp'};
        else
            % Perseveration-based response model
            % 4 parameters: [nu, kappa, omega, rho]
            lb = [1e-4  0   -5  -10];
            ub = [2   2   +5      10];
            % 
            param_names = {'nu', 'kappa', 'omega','rho'};
        end
        
        config = struct('bound', [lb; ub]);

        parameters    = nan(N, numel(lb));
        loglik        = nan(N, 1);
        lme           = nan(N, 1);

        signals       = cell(N, 1);    
        accuracy      = cell(N, 1);

        for n=1:N        
            model_fun = @(params, d) model_4fit(params, d, experiment);
            [parameters(n, :), loglik(n), lme(n)] = tools_fit(data{n}, model_fun, config);

            [~, signals{n}, accuracy{n}] = model_4fit(parameters(n, :), data{n}, experiment);
            fprintf('Subject %03d/%03d\n', n, N);
        end
            
        meta = struct();
        meta.experiment = experiment;
        meta.bounds_lb = lb;
        meta.bounds_ub = ub;

        save(fname, 'parameters', 'loglik', 'lme', 'signals', 'param_names', 'accuracy', 'meta');    
    end
    
    % Summarize fitted parameters 
    f = load(fname);
    parameters  = f.parameters;
    param_names = f.param_names;

    % ---- parameter percentiles ----
    x = prctile(parameters, [25 50 75], 1)'; 
    tbl.data    = x;
    tbl.rows    = param_names;
    tbl.columns = {'25%','50%','75%'};

    T = array2table(tbl.data, ...
        'VariableNames', tbl.columns, ...
        'RowNames', tbl.rows);
    fprintf('\n=== Fitted parameter summary (percentiles) ===\n');
    disp(T);

    % ---- learning-rate summary ----
    % exclude bad_traj; try use median
    dynamics    = f.signals;
    N = numel(dynamics);
    B = size(dynamics{1}.LR2, 2);
    lr_blockmean = nan(N, B);
    bad_count = 0;

    for n = 1:N
        if dynamics{n}.bad_traj == 0
            lr_blockmean(n,:) = median(dynamics{n}.LR2, 1, 'omitnan');
        else
            bad_count = bad_count + 1;
            lr_blockmean(n,:) = nan;
        end
    end
    fprintf('Bad trajectory %d / %d\n', bad_count, length(lr_blockmean));
    
    mean_alpha = mean(lr_blockmean, 1, 'omitnan');
    sem_alpha  = std(lr_blockmean, 0, 1, 'omitnan') ./ sqrt(sum(~isnan(lr_blockmean),1)); 
    fprintf('Mean alpha per block: %s\n', mat2str(mean_alpha, 4));
    fprintf('SEM alpha per block:  %s\n', mat2str(sem_alpha, 4));
    lr_sub = lr_blockmean;


    % Add mean_ and median_dynamics using mean and median of fitted params
    bad_traj = parameters(:, 3)==0;
    mean_params   = mean(parameters(~bad_traj, :), 1, 'omitnan');
    median_params = median(parameters(~bad_traj, :), 1, 'omitnan');
    
    % ---- compute group-level dynamics using mean / median params ----  
    fprintf('Computing mean-parameter binary HGF dynamics...\n');
    [~, mean_signals, ~] = model_4fit(mean_params, data{1}, experiment);
    fprintf('Computing median-parameter binary HGF dynamics...\n');
    [~, median_signals, ~] = model_4fit(median_params, data{1}, experiment);

    fprintf("Aggregate and save results.\n");
    mean_params_glm_lr = mean_signals.glm_lr;
    median_params_glm_lr = median_signals.glm_lr;

    save(fname, '-append', 'mean_alpha', 'sem_alpha', 'lr_sub', 'mean_signals', 'median_signals', 'mean_params_glm_lr', 'median_params_glm_lr');
  
end

% -------------------------------------------------------------------------
function [loglik, signals, accuracy] = model_4fit(parameters, data, experiment)
    outcome = data.outcome;   % ground-truth feedback (0/1)
    choice = data.choice;    % subject prediction (0/1)
    outcome(isnan(choice)) = NaN;  % mask outcome when choice missing

    % --- unpack parameters ---
    theta = parameters(1:3);     % parameters for binary_hgf_model (nu, kappa, omega, etc.)

    % forward pass: baseline predicted P(choice==1) from cognitive model
    [p_pred, signals, bad_traj] = binary_hgf_model(theta, outcome);

    % response model: apply perseveration at choice stage + compute likelihood
    % ----- response model switch -----
    if strcmp(experiment, 'binA')
        % bias model
        resp_params = struct( ...
            'bv', parameters(4), ...
            'be', parameters(5), ...
            'bi', parameters(6) ...
        );
        out = response_model_A(p_pred, choice, resp_params);
        % rho_val = NaN;
        % beta_val = NaN;
        % bias_vals = parameters(4:6);
    elseif strcmp(experiment, 'binB')
        % inverse beta temp 
        trialvalue = data.trialvalue;
        resp_params = struct('temp', parameters(4));
        out = response_model_B(p_pred, trialvalue, choice, resp_params);
        % rho_val = NaN;
        % beta_val = parameters(4)
        % bias_vals = [NaN NaN NaN];
    else
        % perseveration model
        resp_params = struct('rho', parameters(4));
        out = response_model(p_pred, choice, resp_params);
        % rho_val = parameters(4);
        % beta_val = NaN;
        % bias_vals = [NaN NaN NaN];
    end
    loglik = out.loglik;

    if bad_traj
        prob = ones(size(choice))/2;
        loglik = sum(log(prob(:)));
    end

    if nargout > 1
        [~, signals, ~] = binary_hgf_model(theta, outcome);
    end

    % accuracy
    stats = size(outcome,1);
    t1 = 1:(stats-1);
    accuracy = (choice == outcome);
    accuracy = accuracy(t1,:);
end
