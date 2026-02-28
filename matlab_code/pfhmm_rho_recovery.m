function st = pfhmm_rho_recovery(doPlot)
    if nargin < 1, doPlot = true; end
    close all;

    %% Setup paths
    currentFolder = fileparts(mfilename('fullpath'));
    cd(currentFolder);
    addpath('cbm');

    experiment = 'sealion';
    [data] = get_data(experiment);

    % Simulation file name
    experiment = 'sim';
    postfix = '';
    fdir = fullfile('..', 'mat_data', sprintf('experiment_%s', experiment));
    simfile = fullfile(fdir, sprintf('data_sim_binary_pfhmm_rho%s.mat', postfix));
    fprintf("Using simulated data file: %s\n", simfile);

    % PFHMM + BayesOpt configuration (must match fitting)
    config = struct('num_inits', 10);
    pf_cfg = struct();
    pf_cfg.s0 = 0.25;
    pf_cfg.v0 = 0.25;
    pf_cfg.resampling_strategy = 'systematic';
    pf_cfg.resample_percentage = 0.5;
    pf_cfg.num_particles = 10000;
    pf_cfg.response_model = 'rho';   % recovery for rho-based PFHMM
    pf_cfg.experiment = experiment;
    model = @(x, dat, mode) pfhmm_rho(x, dat, mode, pf_cfg);

    %% Simulate PFHMM data
    if ~exist(simfile,'file')
        fprintf('Simulating PFHMM recovery data → %s\n', simfile);

        % --- get parameter sampler from PFHMM ---
        [~, ~, vars] = model([], [], 'config');
        pnames    = vars.parameters_name;
        rand_func = vars.rand_func;

        N = 100;
        sim_data = cell(N,1);
        true_parameters = nan(N, numel(pnames));
        outcome = data{1}.outcome;

        for n = 1:N
            fprintf('Sim subject %d/%d\n', n, N);
            
            % sample true parameters (RAW space)
            rng(n);
            tp = rand_func(1);
            tp_tbl = array2table(tp, 'VariableNames', pnames);
            true_parameters(n,:) = tp;

            % average PF forward passes (beliefs only)
            val_mean = 0;
            for k = 1:config.num_inits
                [~, val] = model(tp_tbl, struct('outcome', outcome), 'sim');
                val_mean = val_mean + val / config.num_inits;
            end
            
            % generate choices sequentially so rho is active
            [T,B] = size(val_mean);
            choice = nan(T,B);
            
            for t = 1:T
                if t == 1
                    prev = nan(1,B);
                else
                    prev = choice(t-1,:);
                end
            
                resp = response_model( ...
                    val_mean(t,:), ...
                    prev, ...
                    struct('rho', tp_tbl.rho) ...
                );
            
                choice(t,:) = binornd(1, resp.p1);
            end

            sim_data{n} = struct( ...
                'outcome', outcome, ...
                'choice',  choice ...
            );
        end

        save(simfile, 'sim_data', 'true_parameters', 'pnames');
    end

    %% Fit PFHMM to simulated data
    fitfile = fullfile(fdir, sprintf('pfhmm_rho_fit_%s%s.mat', experiment, postfix));
    fprintf('Running pfhmm_fit on simulated data...\noutput to: %s\n', fitfile);
    pfhmm_rho_fit('sim', postfix);

    %% Load fitted parameters
    f_sim = load(simfile);
    true_parameters = f_sim.true_parameters;
    pnames = f_sim.pnames;

    f_fit = load(fitfile);
    parameters = f_fit.parameters;   % cell array of tables

    N = size(true_parameters,1);
    P = numel(pnames);

    fitted_parameters = nan(N,P);
    for n = 1:N
        fitted_parameters(n,:) = table2array(parameters{n});
    end

    %% SECTION 4: Recovery analysis
    st = struct();
    st.param_names = pnames;

    % Compute fitting error and its quantiles
    error = true_parameters - fitted_parameters;
    q25 = quantile(error, 0.25);
    q50 = quantile(error, 0.50);
    q75 = quantile(error, 0.75);
    %   Row 1: 25% quantile, Row 3: median, Row 4: 75% quantile.
    statsMatrix = [q25; q50; q75];
    % Transpose it so that each row corresponds to a specific parameter.
    statsMatrix = statsMatrix';
    % Define row names: first 4 rows for Volatility (Block1 to Block4) and next 4 for Stochasticity.
    rowNames = {'$\sigma_{s}$', '$\sigma_{v}$', '$\rho$'};
    % Define column names for the statistics.
    colNames = {'25% quantile','Median','75% quantile'};
    
    st.table.data = statsMatrix;
    st.table.rows = rowNames;
    st.table.columns = colNames;

    % T = array2table(st.table.data, 'RowNames', st.table.rows, 'VariableNames', st.table.columns);
    % disp(T);

    %% SECTION 5: Plots
    if doPlot
        plot_recovery(error, rowNames);
        saveas(gcf,'../saved_figures/SuppFigure4_pfhmm_rho_recovery.png','png');
    end


end

% Plot recovery error distributions
function plot_recovery(error, pnames)

    nr = 3; nc = 3;
    figure('units','normalized','position',[0.1 0.1 0.7 0.6]);

    fsy = 14;
    fsalpha = 18;
    col = [.5 .2 .1];
    alf = .2;
    
    % xmax = max(abs(error(:)));
    % xmax = ceil(xmax*10)/10;
    xmax = 0.4;

    for i = 1:size(error,2)
        subplot(nr,nc,i);

        [fq,xq] = ksdensity(error(:,i));
        plot(xq,fq,'r-','LineWidth',1.5); hold on;
        if i==3
            xlim([-1 1]);
            ylim([0 2.5]);
        else
            xlim([-xmax xmax]);
            ylim([0 5]);
        end

        yl = ylim;
        xlabel('Fitting error','fontsize',fsy);
        ylabel('Density','fontsize',fsy);
        title(pnames{i},'Interpreter','latex','fontsize',fsalpha);

        mx = mean(error(:,i));
        sx = std(error(:,i));
        fill([mx-sx mx+sx mx+sx mx-sx], ...
             [yl(1) yl(1) yl(2) yl(2)], ...
             col,'FaceAlpha',alf,'EdgeColor','none');

        plot([mx mx],yl,'r--','LineWidth',1);
        plot([0 0],yl,'k:','LineWidth',1);
        ylim(yl);
    end
end