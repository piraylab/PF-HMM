function [tbl] = pearcehall_fmincon(experiment)
% PEARCEHALL_FMINCON  Fit Pearce-Hall + response_model using fmincon (tools_fit)
    if nargin < 1
        experiment = 'sealion'; 
        % 'sealion', 'turtle', 'binA', 'binB'
        postfix = ''; 
    end

    addpath("tools");
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

    % save file name
    fname = fullfile(fdir, sprintf('%s_%s%s.mat', mfilename, experiment, postfix));

    if ~exist(fname, 'file')
        % --- bounds in RAW space (since pearcehall_model transforms internally) ---
        if strcmp(experiment, 'binA')
            % Bias-based response model
            % parameters: [kappa_raw, bv, be, bi]
            lb = [0,  -10,  -10,  -10];
            ub = [1,   10,   10,   10];
            param_names = {'kappa','bv','be','bi'};
        elseif strcmp(experiment, 'binB')
            % Beta-based response model
            % parameters: [kappa, temp]
            lb = [0,  -10];
            ub = [1,   10];
            param_names = {'kappa','temp'};
        else
            % Perseveration-based response model
            % parameters: [kappa_raw, rho]
            lb = [0,  -10];
            ub = [1,   10];
            param_names = {'kappa', 'rho'};
        end
        
        config = struct('bound', [lb; ub]);

        N = numel(data);
        blocks = size(data{1}.outcome, 2);

        parameters    = nan(N, numel(lb));
        loglik        = nan(N, 1);
        lme           = nan(N, 1);

        tx            = nan(N, 3);    % adjust if your pearcehall_model returns different length
        glm_lr        = nan(N, blocks);
        signals       = cell(N, 1);
        accuracy      = cell(N, 1);

        for n=1:N        
            model_fun = @(params, d) model_4fit(params, d, experiment);
            [parameters(n, :), loglik(n), lme(n)] = tools_fit(data{n}, model_fun, config);            
            [~, tx(n,:), glm_lr(n,:), signals{n}, accuracy{n}] = model_4fit(parameters(n, :), data{n}, experiment);
            fprintf('Subject %03d/%03d\n', n, N);
        end

        meta = struct();
        meta.experiment = experiment;
        meta.bounds_lb = lb;
        meta.bounds_ub = ub;

        alpha_summary = summarize_PH_alpha(signals); 

        save(fname, 'parameters', 'loglik', 'lme', 'tx', 'glm_lr', 'alpha_summary', 'signals', 'param_names', 'accuracy', 'meta');
        fprintf("Saved: %s\n", fname);
    end

    % Summarize fitted parameters 
    f = load(fname);
    parameters = f.parameters;
    param_names = f.param_names;
    
    % ---- parameter percentiles ----
    x = prctile(parameters, [25 50 75], 1)';   % P x 3
    tbl.data    = x;
    tbl.rows    = param_names;
    tbl.columns = {'25%','50%','75%'};
    
    T = array2table(tbl.data, ...
        'VariableNames', tbl.columns, ...
        'RowNames', tbl.rows);
    % fprintf('\n=== Fitted parameter summary (percentiles) ===\n');
    % disp(T);
    
    % ---- learning-rate summary ----
    alpha_summary = f.alpha_summary;
    m_alpha = alpha_summary.mean_alpha_per_block;
    sem_alpha = alpha_summary.sem_alpha_per_block;
    B = numel(m_alpha);
    fprintf('Learning rate (mean ± SEM per block):\n');
    for b = 1:B
        fprintf('  Block %d: %.4f ± %.4f\n', b, m_alpha(b), sem_alpha(b));
    end

    % Add mean_ and median_dynamics using mean and median of fitted params
    mean_params   = mean(parameters, 1, 'omitnan');
    median_params = median(parameters, 1, 'omitnan');
    
    % ---- compute group-level dynamics using mean / median params ----  
    fprintf('Computing mean-parameter PFHMM dynamics...\n');
    [~, ~, mean_params_glm_lr, ~, ~] = model_4fit(mean_params, data{1}, experiment);
    fprintf('Computing median-parameter PFHMM dynamics...\n');
    [~, ~, median_params_glm_lr, ~, ~] = model_4fit(median_params, data{1}, experiment);

    save(fname, '-append', 'mean_params_glm_lr', 'median_params_glm_lr');
  
end
% -------------------------------------------------------------------------
function [loglik, tx, lr, signals, accuracy] = model_4fit(parameters, data, experiment)
    outcome = data.outcome;   % ground-truth feedback (0/1)
    choice = data.choice;    % subject prediction (0/1)
    outcome(isnan(choice)) = NaN;    % mask outcome when choice missing 

    kappa_raw = parameters(1);     % parameters for Pearce-Hall model

    % ----- forward pass: cognitive model -----
    p_pred = pearcehall_model(kappa_raw, outcome, 0); % fmincon fitting so no transformation needed

    % ----- response model switch -----
    
    if strcmp(experiment, 'binA')
        % bias model
        resp_params = struct( ...
            'bv', parameters(2), ...
            'be', parameters(3), ...
            'bi', parameters(4) ...
        );
        out = response_model_A(p_pred, choice, resp_params);
    elseif strcmp(experiment, 'binB')
        % inverse beta temp 
        trialvalue = data.trialvalue;
        resp_params = struct('temp', parameters(2));
        out = response_model_B(p_pred, trialvalue, choice, resp_params);
    else
        % perseveration model
        resp_params = struct('rho', parameters(2));
        out = response_model(p_pred, choice, resp_params);
    end
    loglik = out.loglik;

    % ----- return internal signals if requested -----
    if nargout > 1
        [~, tx, lr, signals] = pearcehall_model(kappa_raw, outcome, 0);
    end

    % ----- accuracy -----
    stats = size(outcome,1);
    t1 = 1:(stats-1);
    accuracy = (choice == outcome);
    accuracy = accuracy(t1,:);
end