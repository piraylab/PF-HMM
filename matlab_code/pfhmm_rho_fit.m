function st = pfhmm_rho_fit(experiment, postfix)
    if nargin < 2
        experiment = 'sealion'; % 'sealion', 'turtle', 'binA', 'binB'
        postfix = ''; 
    end
    addpath("tools");

    fdir = fullfile('..', 'mat_data', sprintf('experiment_%s', experiment));
    if ~exist(fdir,'dir'), error("Experiment folder not found: %s", fdir); end
    
    if strcmpi(experiment, 'sim')
        f = load(fullfile(fdir, sprintf("data_sim_binary_pfhmm_rho%s.mat", postfix)));
        data = f.sim_data;
    elseif strcmpi(experiment, 'binA') || strcmpi(experiment, 'binB')
        data = get_data_bin(experiment);
    else
        [data] = get_data(experiment);
    end

    % save file name
    fname = fullfile(fdir, sprintf('%s_%s%s.mat', mfilename, experiment, postfix));

    if ~exist(fname, 'file')
        fprintf("Creating result file%s: \n", fname);
        % bayesopt config
        config = struct('num_bayespoint', 200, 'num_seeds', 5, 'seed', 0, 'num_inits', 10);    
        
        % PF config (must include s0/v0 AND PF settings)
        pf_cfg = struct();
        pf_cfg.s0 = 0.25;   % run initialization with constant values 
        pf_cfg.v0 = 0.25;   
        % ---------------- response model selection ----------------
        pf_cfg.experiment = experiment;

        % Model handle: keep signature (x, dat, mode)
        model = @(x, dat, mode) pfhmm_rho(x, dat, mode, pf_cfg);
    
        % Config for bayesopt variables
        [~, ~, vars] = model([], [], 'config');
        opt_var  = vars.opt_var;
        InitialX = vars.InitialX;    
    
        % Preallocate results
        Nsub = length(data);
        loglik     = nan(Nsub, 1);
        parameters = cell(Nsub, 1);
        obs_dynamics   = cell(Nsub, 1);
        est_dynamics = cell(Nsub, 1);
    
        % Fit directory
        fit_dir = fullfile(fdir, sprintf('%s_%s%s', mfilename, experiment, postfix));
        if ~exist(fit_dir, 'dir')
            mkdir(fit_dir);
        end
    
        % Fit each subject
        for n=1:Nsub
            fname_subj = fullfile(fit_dir, sprintf('subj_%04d.mat', n));
            
            if ~exist(fname_subj, 'file')
                fprintf("Fitting subject %d: ", n);
                dat = data{n};     % contain .outcome and .choice         
                fit_model(fname_subj, config, dat, model, opt_var, InitialX);
            end    
            f = load(fname_subj);    
            loglik(n) = f.pf_observed.loglik;   % scalar (mean over inits)
            parameters{n} = f.pf_observed.x;    % table row (best params)
    
            obs_dyn = f.pf_observed.signal.dynamics;
            obs_dynamics{n} = obs_dyn;

            est_dyn = f.pf_estimated.signal.dynamics;
            est_dynamics{n} = est_dyn;
        end
        fprintf("Per subject fitting done!\n");

        % LME / BIC-style penalty (per subject)
        dat0 = data{1};
        T = numel(~isnan(dat0.outcome(:)));
        num_parameters = width(InitialX);
        
        lme = loglik - 0.5*num_parameters*log(T);

        % Add mean_ and median_dynamics using mean and median of fitted params
        pnames = parameters{1}.Properties.VariableNames;
        P = numel(pnames);
        param_mat = nan(Nsub, P);
        for n = 1:Nsub
            param_mat(n,:) = table2array(parameters{n});
        end
        mean_params   = mean(param_mat, 1, 'omitnan');
        median_params = median(param_mat, 1, 'omitnan');
        
        mean_params_tbl   = array2table(mean_params,   'VariableNames', pnames);
        median_params_tbl = array2table(median_params, 'VariableNames', pnames);
        % ---- compute group-level dynamics using mean / median params ----    
        num_pf_runs = config.num_inits; % same as above fitting
        fprintf('Computing mean-parameter PFHMM dynamics...\n');
        mean_dynamics = make_signal(mean_params_tbl, model, data{1}, num_pf_runs);
        mean_dynamics = mean_dynamics.dynamics;
        
        fprintf('Computing median-parameter PFHMM dynamics...\n');
        median_dynamics = make_signal(median_params_tbl, model, data{1}, num_pf_runs);
        median_dynamics = median_dynamics.dynamics;

        fprintf("Aggregate and save results.\n");
        save(fname, ...
             'loglik', 'parameters', ...
             'obs_dynamics', 'est_dynamics', ...
             'mean_dynamics', 'median_dynamics', ...
             'mean_params_tbl', 'median_params_tbl', ...
             'lme', 'config', 'pf_cfg');
        
        % ---- post-hoc block-effect estimation ----
        if strcmpi(experiment,'turtle_pilot') || strcmpi(experiment,'sealion_pilot')
            add_block_effect_to_pfhmm(experiment, postfix,'rho_');
        end

    end
    
    % PFHMM summary: parameters + learning rate
    f = load(fname);
    
    parameters   = f.parameters;      % cell array (Nsub x 1), each a table
    obs_dynamics = f.obs_dynamics;    % cell array (Nsub x 1)
    
    Nsub = numel(parameters);
    
    % 1) Fitted parameter summary (percentiles)
    st.st_params = struct();
    % Extract parameter names from first subject
    pnames= parameters{1}.Properties.VariableNames;
    st.st_params.rows = pnames;
    
    P = numel(pnames);
    param_mat = nan(Nsub, P);
    for n = 1:Nsub
        param_mat(n,:) = table2array(parameters{n});
    end
    % Percentiles
    st.st_params.data = prctile(param_mat, [25 50 75], 1)';   % P x 3
    st.st_params.columns = {'25%','50%','75%'};

    st.st_params.table = array2table( ...
    st.st_params.data, ...
    'VariableNames', st.st_params.columns, ...
    'RowNames', st.st_params.rows ...
    );
    
    % fprintf('\n=== PFHMM fitted parameter summary (percentiles) ===\n');
    % disp(st.st_params.table);

    % 2) Learning-rate (alpha) summary per block
    st.st_lr = struct();
    % PFHMM internal LR stored as obs_dynamics{n}.lr (1 x B)
    B = numel(obs_dynamics{1}.lr);
    lr_sub = nan(Nsub, B);
    for n = 1:Nsub
        lr_sub(n,:) = obs_dynamics{n}.lr;
    end
    
    mean_lr = mean(lr_sub, 1, 'omitnan');
    sem_lr  = std(lr_sub, 0, 1, 'omitnan') ./ ...
              sqrt(sum(~isnan(lr_sub),1));
    st.st_lr.mean = mean_lr;      % 1 x B
    st.st_lr.sem  = sem_lr;       % 1 x B
    st.st_lr.data = [mean_lr(:), sem_lr(:)];   % B x 2
    st.st_lr.rows = arrayfun(@(b) sprintf('Block%d', b), 1:B, 'UniformOutput', false);
    st.st_lr.columns = {'Mean_alpha','SEM_alpha'};

    st.st_lr.table = array2table( ...
        st.st_lr.data, ...
        'VariableNames', st.st_lr.columns, ...
        'RowNames', st.st_lr.rows ...
    );
    
    % fprintf('\n=== PFHMM learning rate (mean ± SEM per block) ===\n');
    % disp(st.st_lr.table);
    
    % 3) (Optional) Block-effect GLM summary
    if isfield(obs_dynamics{1}, 'block_effect')
        block_eff_sub = nan(Nsub, B);
        for n = 1:Nsub
            block_eff_sub(n,:) = obs_dynamics{n}.block_effect;
        end

        mean_be = mean(block_eff_sub, 1, 'omitnan');
        sem_be  = std(block_eff_sub, 0, 1, 'omitnan') ./ ...
                  sqrt(sum(~isnan(block_eff_sub),1));

        tbl_blockeff = table( ...
            mean_be', sem_be', ...
            'VariableNames', {'Mean_block_effect','SEM_block_effect'}, ...
            'RowNames', st.st_lr.rows ...
        );

        % fprintf('\n=== PFHMM block-effect GLM estimates (mean ± SEM) ===\n');
        % disp(tbl_blockeff);

        lr_pfhmm = lr_sub;  
        block_eff = block_eff_sub;
        summary_fname = fullfile(fdir, sprintf('pfhmm_rho_lr_%s%s.mat', experiment, postfix));
        median_params_lr = f.median_dynamics.lr;
        mean_params_lr = f.mean_dynamics.lr;
        save(summary_fname, 'lr_pfhmm', 'block_eff', 'median_params_lr', 'mean_params_lr');
    end
   
end

% -------------------------------------------------------------------------
function bayesopt_results = fit_model(fname, config, dat, model, opt_var, InitialX)

    % bayesopt MINIMIZES. We want to MAXIMIZE loglik => minimize (-loglik).
    opt_fun = @(x) optimization(x, model, dat, config.num_inits, 'rmean');

    rng(config.seed);
    bayesopt_results = bayesopt(opt_fun, opt_var, ...
        'IsObjectiveDeterministic', false, ...
        'Verbose', 1, ...
        'PlotFcn', {}, ...
        'AcquisitionFunctionName', 'expected-improvement-plus', ...
        'InitialX', InitialX, ...
        'MaxObjectiveEvaluations', config.num_bayespoint, ...
        'NumSeedPoints', config.num_seeds);

    [pf_observed, pf_estimated] = fit_post(bayesopt_results, dat, model, config);

    if ~isempty(fname)
        save(fname, 'bayesopt_results', 'config', 'pf_estimated', 'pf_observed');
    end
end

% -------------------------------------------------------------------------
function [pf_observed, pf_estimated] = fit_post(bayesopt_results, dat, model, config)

    % Evaluate mean loglik at the two bayesopt optima
    opt_eval = @(x) optimization(x, model, dat, config.num_inits, 'mean');

    rng(config.seed);
    [~, ~, mean_loglik, loglik_runs] = opt_eval(bayesopt_results.XAtMinObjective);
    pf_observed = struct( ...
        'x', bayesopt_results.XAtMinObjective, ...
        'loglik', mean_loglik, ...
        'loglik_runs', loglik_runs);
    rng(config.seed);
    pf_observed.signal = make_signal(pf_observed.x, model, dat, config.num_inits);

    rng(config.seed);
    [~, ~, mean_loglik, loglik_runs] = opt_eval(bayesopt_results.XAtMinEstimatedObjective);
    pf_estimated = struct( ...
        'x', bayesopt_results.XAtMinEstimatedObjective, ...
        'loglik', mean_loglik, ...
        'loglik_runs', loglik_runs);
    rng(config.seed);
    pf_estimated.signal = make_signal(pf_estimated.x, model, dat, config.num_inits);
end

% -------------------------------------------------------------------------
function [objective, c, mean_loglik, nloglik_runs] = optimization(x, model, dat, num_init, obj_stats)
%OPTIMIZATION  Aggregate stochastic PFHMM log-likelihoods across runs.
%
% Returns:
%   mean_loglik : mean loglik across runs
%   objective   : value MINIMIZED by bayesopt (negative of aggregated loglik)

    nloglik_runs = nan(1, num_init);

    for k = 1:num_init
        % pfhmm returns scalar loglik
        [ll, ~, ~] = model(x, dat, 'fit');
        nloglik_runs(k) = -ll;
    end

    if strcmp(obj_stats, 'mean')
        objective = mean(nloglik_runs);
    elseif strcmp(obj_stats, 'rmean')
        objective = mean(nloglik_runs) + std(nloglik_runs)/sqrt(num_init);  % optimistic? or conservative? see note below
    else
        error('unknown obj_stats!');
    end

    c = [];
    mean_loglik = mean(-nloglik_runs);
end

% -------------------------------------------------------------------------
function sig = make_signal(parameters, model, dat, num_inits)
%MAKE_SIGNAL  Average PFHMM latent signals over stochastic runs.

    for i = 1:num_inits
        % Use 'sim' so pfhmm returns vars
        [~, ~, vars_i] = model(parameters, dat, 'sim');

        snames = fieldnames(vars_i);
        for j = 1:length(snames)
            if i == 1
                dynamics.(snames{j}) = 0;
            end
            dynamics.(snames{j}) = dynamics.(snames{j}) + vars_i.(snames{j}) / num_inits;
        end
    end

    sig = struct('dynamics', dynamics, 'parameters', parameters);
end