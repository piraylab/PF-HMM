function hgf_plot

    experiment = 'sealion'; 

    postfix = ''; 
    fdir = fullfile('..', 'mat_data', sprintf('experiment_%s', experiment));
    fname = fullfile(fdir, 'hidden_state.mat');
    f = load(fname);    
    timeseries = f.timeseries;
    outcome = timeseries.observations;
    state = timeseries.hidden_state;

    fname = fullfile(fdir, sprintf('binary_hgf_fit_%s%s.mat', experiment, postfix));
    f = load(fname);    

    parameters = f.parameters;
    bad_traj = parameters(:, 3)==0;

    tx{1} = median(parameters(~bad_traj, :));
    tx{2} = median(parameters(~bad_traj, :));
    tx{2}(2) = 0; % no volatility

    outcome = repmat(outcome(:, 1), 1, 2);
    state = repmat(state(:, 1), 1, 2);

    do = [state(:,1)~=outcome(:, 1)]~=0;    
    ds = [0; diff(state(:,1))]~=0;
    do = find(do);
    ds = find(ds);


    for i=1:2
        [pred(:, i), out, bad_traj] = binary_hgf_model(tx{i}, outcome(:, i));
    
        lr(:, i) = out.trial_lr;
        vol(:, i) = out.vol;
    end
    delta = (outcome - pred); delta(end, :) = [];
    update = pred(2:end, :) - pred(1:end-1, :);

    [T, B] = size(lr);
    close all;
    figure('Position',[100 100 750 800]);

    ylim_delta = [min(delta(:)); max(delta(:))].*[1.05; 1.05];
    ylim_vol = [min(vol(:)); max(vol(:))].*[.95; 1.05];
    ylim_lr = [min(lr(:)); max(lr(:))].*[.95; 1.05];
    labels = {'HGF', 'HGF with fixed volatility'};
    
    for b = 1:B

        subplot(4,B,b)
        plot(1:T, outcome(1:T,b),'x','LineWidth',1.5); hold on;
        plot(1:T, state(1:T,b),'k', 'LineWidth',1.5); hold on;
        ylim([-.49 1.49])

        title(sprintf(labels{b}))
        ylabel('Observations')
        
%         grid on        
%         plot_lines(do, ds)

        subplot(4,B,b+B)
        plot(1:T, delta(1:T,b),'k','LineWidth',1.5)
        ylabel('Prediction errors')
        ylim(ylim_delta)
        plot_lines(do, ds)

        subplot(4,B,b+2*B)
        
        plot(1:T, vol(1:T,b),'k','LineWidth',1.5)
        xlabel('Trial')
        ylabel('Volatility')
        ylim(ylim_vol)
        plot_lines(do, ds)
%         grid on

        

        % Row 2: Level-2 variance
        subplot(4,B,b+3*B)
        plot_lines(do, ds)
        plot(1:T, lr(1:T,b),'k','LineWidth',1.5)
        xlabel('Trial')
        ylabel('Learning rate')
        ylim(ylim_lr)
%         grid on

        saveas(gcf,'../saved_figures/SuppFigure1_hgf_lr.png','png');
    
    end
end

function plot_lines(do, ds)
do = [do; ds];
yl = get(gca, 'ylim'); hold on;
colr = repmat({[70, 130, 180]/255}, size(do));
colr(do==ds) = { [139, 0, 0]/255};

for i=1:length(do)
    h = plot(do(i)*[1; 1], yl, 'Color', colr{i}, 'linewidth', 1); hold on;
%     h.Color(4) = 0.8;
end
set(gca, 'ylim', yl);
set(gcf, 'Renderer', 'opengl');
end