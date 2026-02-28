function st = hmm_rho_recovery(doPlot)
% HMM_RECOVERY  Parameter recovery for HMM + response model (rho)
%
% Recovers:
%   - Volatility (4 blocks)
%   - Stochasticity (4 blocks)
%   - Perseveration rho (global)
%
% Uses response_model to generate choices during simulation.

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

    % Simulation
    N = 100;
    sim_data = cell(N,1);
    fname = fullfile(fdir, sprintf('data_sim_binary_hmm_rho%s.mat', postfix));
    fprintf("Using simulated data file: %s\n", fname);

    if ~exist(fname,'file')

        fprintf('Simulating HMM recovery data...\n');
        % Import fitted CBM parameters from real sealion task
        % (use mean and std in RAW parameter space)
        real_experiment = 'sealion';
        real_postfix = postfix;
        real_fdir = fullfile(fdir, sprintf('experiment_%s', real_experiment));
        real_fname = fullfile(real_fdir, ...
            sprintf('hmm_rho_fit_%s%s.mat', real_experiment, real_postfix));
        f_real = load(real_fname);
        cbm_real = f_real.cbm;

        % Extract raw fitted parameters (before transformation)
        % Each subject has 9 params: 4 vol, 4 sto, 1 rho
        Nreal = numel(cbm_real);
        raw_params = zeros(Nreal, 9);

        for i = 1:Nreal
            raw_params(i,:) = cbm_real{i}.output.parameters(:)';
        end

        % Compute empirical mean and std in RAW space
        mu_params = mean(raw_params, 1);
        sd_params = std(raw_params, 0, 1);

        % Simulate new subjects using empirical parameter distribution
        for n = 1:N
            rng(n);

            % Sample raw parameters from Normal(mu, sd)
            theta_raw = mu_params + sd_params .* randn(1,9);

            % Convert to simulation parameter struct
            % (apply same transformation as model2fit)
            half_sigmoid = @(x) 0.5 ./ (1 + exp(-x));

            sim_vol = half_sigmoid(theta_raw(1:4));
            sim_sto = half_sigmoid(theta_raw(5:8));
            sim_rho = theta_raw(9);

            sim_params = struct( ...
                'sim_vol', sim_vol, ...
                'sim_sto', sim_sto, ...
                'rho',     sim_rho ...
            );

            % Simulate outcome + choice
            config = struct();
            config.seed = n;
            config.outcome = data{1}.outcome;
            config.sim_params = sim_params;

            sim_data{n} = sim_outcome_choice(config);
        end

        save(fname,'sim_data');

    end
    
    % Fit HMM to simulated data
    hmm_rho_fit(experiment, postfix);

    % Extract true parameters
    f = load(fname);
    sim_data = f.sim_data;
    nSim = numel(sim_data);

    % true parameters: [vol(1:4), sto(1:4), rho]
    sim_params = zeros(nSim, 9);
    for i = 1:nSim
        sim_params(i,:) = [ ...
            sim_data{i}.sim_params.sim_vol, ...
            sim_data{i}.sim_params.sim_sto, ...
            sim_data{i}.sim_params.rho ...
        ];
    end

    % Load recovered parameters
    fname = sprintf('hmm_rho_params_sim%s.mat', postfix);
    f = load(fullfile(fdir,fname));
    fprintf("Loading recovered parameters file: %s\n", fname);
    hmm_params = f.parameters;   % N x 12

    % Recovery error
    error = sim_params - hmm_params;

    q25 = quantile(error, 0.25);
    q50 = quantile(error, 0.50);
    q75 = quantile(error, 0.75);

    statsMatrix = [q25; q50; q75]';
    
    rowNames = { ...
        'Volatility_{Block1}','Volatility_{Block2}', ...
        'Volatility_{Block3}','Volatility_{Block4}', ...
        'Stochasticity_{Block1}','Stochasticity_{Block2}', ...
        'Stochasticity_{Block3}','Stochasticity_{Block4}', ...
        'Rho' ...
    };
    colNames = {'25% quantile','Median','75% quantile'};

    st.table.data = statsMatrix;
    st.table.rows = rowNames;
    st.table.columns = colNames;

    pnames = {'$v_1$','$v_2$','$v_3$','$v_4$', ...
                  '$s_1$','$s_2$','$s_3$','$s_4$', ...
                  '$\rho$'};

    % Plot recovery
    if doPlot
        plot_recovery(error, pnames);
    end
end

% Plot recovery error distributions
function plot_recovery(error, pnames)
    nr = 3; nc = 4;
    figure('units','normalized','position',[0.1 0.1 0.7 0.6]);

    fsy = 14;
    fsalpha = 18;
    col = [.5 .2 .1];
    alf = .2;
    
    % xmax = max(abs(error(:)));
    % xmax = ceil(xmax*10)/10;
    xmax = 0.5;

    for i = 1:size(error,2)
        subplot(nr,nc,i);

        [fq,xq] = ksdensity(error(:,i));
        plot(xq,fq,'r-','LineWidth',1.5); hold on;
        if i==9
            xlim([-2.5 2.5]);
            ylim([0 1.3]);
        else
            xlim([-xmax xmax]);
            ylim([0 10]);
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

    saveas(gcf,'../saved_figures/SuppFigure3_hmm_rho_recovery.png','png');
end

% Simulate choices using HMM + response model (rho)
function data = sim_outcome_choice(config)

    sim_params = config.sim_params;
    outcome = config.outcome;
    rng(config.seed);

    vol = sim_params.sim_vol;
    sto = sim_params.sim_sto;
    rho = sim_params.rho;

    % HMM belief update
    [predictions, ~] = hmm(outcome, vol, sto);

    % Generate choices sequentially so rho is active
    [T,B] = size(predictions);
    choice = nan(T,B);

    for t = 1:T
        if t == 1
            prev = nan(1,B);
        else
            prev = choice(t-1,:);
        end

        resp = response_model( ...
            predictions(t,:), ...
            prev, ...
            struct('rho', rho) ...
        );

        choice(t,:) = binornd(1, resp.p1);
    end

    data = struct( ...
        'outcome', outcome, ...
        'choice',  choice, ...
        'sim_params', sim_params ...
    );
end


