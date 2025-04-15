function st = hmm_recovery(plot)
    %% Setup working environment and paths
    % Get the directory of this file and set it as the current working directory.
    currentFolder = fileparts(mfilename('fullpath'));
    cd(currentFolder);
    % Add the 'cbm' folder (which contains helper functions) to the MATLAB path.
    addpath('cbm');
    % Define the directory where simulation results will be saved.
    fdir = fullfile('..', 'mat_data');
    addpath(fdir);
    
    %% If no input argument, plot the results by default
    if nargin < 1
            plot = true;
    end
    
    %% Load experimental data and select outcome
    % Retrieve data from the experiment 'sealion'
    [data, ~] = get_data('sealion');
    % Use the outcome from the first subject/cell as the basis for simulation.
    outcome = data{1}.outcome;
    
    %% Run simulations to generate synthetic data
    N = 100;                         % Number of simulation runs/subjects
    sim_data = cell(N,1);            % Preallocate cell array to store simulation results
    fname = fullfile(fdir, 'data_sim.mat');
    % Only run simulations if results file does not already exist.
    if ~exist(fname, 'file')
        for n = 1:N
            % Define simulation configuration parameters for each iteration.
            config.num_trials = length(outcome);  % Set number of trials equal to outcome length.
            config.seed = n+1;                      % Use current simulation index as seed.
            rng(n);  % Reset RNG to ensure reproducibility.
            % Generate random simulation parameters between 0.1 and 0.4 for both sim_sto and sim_vol.
            config.sim_params = struct('sim_sto', rand(1,4)*0.4, ... 
                                       'sim_vol', rand(1,4)*0.4);
            config.outcome = outcome;             % Pass outcome data to simulation.
        
            % Run simulation that produces outcome, a binary choice, and simulation parameters.
            [data] = sim_outcome_choice(config);
            % Store simulation results (outcome, choice, and parameters) in the cell array.
            sim_data{n}.outcome = data.outcome;
            sim_data{n}.choice = data.choice;
            sim_data{n}.sim_params = data.sim_params;
        end
        % Save the simulation results to a MAT-file to avoid repeating the simulations.
        save(fname, 'sim_data');
    end
    
    %% Invoke additional fitting procedure (hmm_fit) with identifier 'sim'
    hmm_fit('sim');
    
    %% Extract simulation parameters from the simulation results
    % Reload the simulation data from the file.
    f = load(fname);
    sim_data = f.sim_data;
    nSim = length(sim_data);
    % Preallocate matrix for parameters; each simulation provides 8 values (4 from sim_vol and 4 from sim_sto).
    sim_params = zeros(nSim, 8);
    % Loop through each simulation result and concatenate the sim_vol and sim_sto vectors.
    for i = 1:nSim
        sim_vol = sim_data{i}.sim_params.sim_vol;
        sim_sto = sim_data{i}.sim_params.sim_sto;
        sim_params(i, :) = [sim_vol, sim_sto];
    end
    
    %% Load the recovered HMM parameters from an external file
    f = load(fullfile(fdir, "hmm_params_sim.mat"));
    hmm_params = f.parameters;
    
    %% Compute fitting error and its quantiles
    error = sim_params - hmm_params;
    
    q25 = quantile(error, 0.25);
    q50 = quantile(error, 0.50);
    q75 = quantile(error, 0.75);
    
    %% Organize statistics into a table
    % Build a matrix where each row corresponds to one statistic and each column to a parameter.
    % Currently, statsMatrix is 4×8: 
    %   Row 1: 25% quantile, Row 3: median, Row 4: 75% quantile.
    statsMatrix = [q25; q50; q75];
    % Transpose it so that each row corresponds to a specific parameter.
    statsMatrix = statsMatrix';
    
    % Define row names: first 4 rows for Volatility (Block1 to Block4) and next 4 for Stochasticity.
    rowNames = {...
        'Volatility_{Block1}', 'Volatility_{Block2}', 'Volatility_{Block3}', 'Volatility_{Block4}', ...
        'Stochasticity_{Block1}', 'Stochasticity_{Block2}', 'Stochasticity_{Block3}', 'Stochasticity_{Block4}'};
    % Define column names for the statistics.
    colNames = {'25% quantile','Median','75% quantile'};
    
    st.table.data = statsMatrix;
    st.table.rows = rowNames;
    st.table.columns = colNames;
    
    %% Display the plot
    if plot
        pnames = {'$v_1$','$v_2$','$v_3$','$v_4$','$s_1$','$s_2$','$s_3$','$s_4$'};
        plot_recovery(error, pnames);
    end
    
end
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot the results
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function plot_recovery(error, pnames)
if 1
    close all;    
    nr = 2;
    nc = 4;
    fsiz = [0.1    0.0800    .7    .4];
    subplots = 1:9;
    figure; set(gcf,'units','normalized'); set(gcf,'position',fsiz);
end

fsy = 16;
fsalpha = 20;

col = [.5 .2 .1];
alf = .2;

% supp = {[0 0.5], [0 0.5], 'positive', 'unbounded'};

for i=1:size(error, 2)
    h(i) = subplot(nr,nc,subplots(i));
    [fq,xq] = ksdensity(error(:,i));    
    plot(xq,fq,'r-'); hold on;
    
    yl=get(gca,'ylim');
    xlabel('Fitting error','fontsize',fsy);
    ylabel('Empirical distribution','fontsize',fsy);
    ht = title(pnames{i},'Interpreter','latex','fontsize',fsalpha);
    
    mxq = mean(error(:,i));
    sxq = std(error(:,i));
    [~,t1] = min( abs(xq - (mxq-sxq) ));
    [~,t2] = min( abs(xq - (mxq+sxq) ));
    
    x = xq(t1:t2);
    x2 = [x, fliplr(x)];
    inBetween = [yl(1)*ones(1,length(x)), yl(2)*ones(1,length(x))];
    fill(x2, inBetween, col(1,:), 'FaceAlpha', alf, 'EdgeColor', col(1,:),'EdgeAlpha', alf); hold on;       
    
    plot([mxq mxq],yl*2,'r-','linewidth',1);
    plot([0 0],yl*2,'k-','linewidth',1);
    set(gca,'ylim',yl);

    set(gca,'xlim', [-.4 .4], 'xtick', -.4:.1:.4);
end

saveas(gcf, '../saved_figures/FigureSupp2_recovery.png', 'png')
end
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Helper function for simulating outcome and choice data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [data] = sim_outcome_choice(config)
    % Use simulation parameters and outcome data from the configuration
    sim_params = config.sim_params;
    outcome = config.outcome;
    rng(config.seed);  % Reset RNG using provided seed for reproducibility

    % Extract simulated volatility and stochasticity vectors from the sim_params struct.
    vol = sim_params.sim_vol;
    sto = sim_params.sim_sto;
    % Compute predictions (as probabilities) using the HMM function.
    [predictions, ~] = hmm(outcome, vol, sto);
    % Convert predicted probabilities into a binary decision (0 or 1) using a binomial draw.
    binom_decision = binornd(1, predictions);
    % Return a structure with the outcome, the decision, and the simulation parameters.
    data = struct('outcome', outcome, 'choice', binom_decision, 'sim_params', sim_params);
end


