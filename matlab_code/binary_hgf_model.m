function [predictions, out, bad_traj] = binary_hgf_model(theta, outcomes)
%BINARY_HGF_MODEL Binary Hierarchical Gaussian Filter (HGF)
%
% Implements a 3-level binary HGF for learning under uncertainty.
%
% Inputs
%   theta    : [nu, kappa, omega_raw]
%              nu        ∈ (0,1)   meta-volatility (level 3)
%              kappa     ∈ (0,1)   coupling between volatility and learning
%              omega     ∈ (-5,0)    baseline log-volatility parameter
%
%   outcomes : N x Q matrix of binary outcomes (0/1),
%              where Q indexes blocks/conditions.
%              NaNs indicate missing feedback and are skipped.
%
% Outputs
%   predictions : N x Q predicted P(outcome = 1)
%
%   out : struct with fields
%       .mu1hat   : N x Q predicted outcome probability
%       .mu2      : (N+1) x Q belief means (level 2)
%       .mu3      : (N+1) x Q volatility states (level 3)
%       .sigma2   : (N+1) x Q posterior variance (level 2)
%       .sigma3   : (N+1) x Q posterior variance (level 3)
%       .LR2      : N x Q learning rate at level 2
%       .bad_traj : true if numerical instability occurred

    % unpack theta
    nu = theta(1);
    kappa = theta(2);
    omega = theta(3);

    [N,Q] = size(outcomes);

    % prepend dummy row 
    y = [zeros(1,Q); outcomes];

    mu2    = nan(N+1,Q);
    mu3    = nan(N+1,Q);
    sigma2 = nan(N+1,Q);
    sigma3 = nan(N+1,Q);
    mu1hat = nan(N+1,Q);

    mu2(1,:)    = 0;
    sigma2(1,:) = 0.1;
    mu3(1,:)    = 1;
    sigma3(1,:) = 1;

    bad_traj = false;
    ux = @(x) 1./(1+exp(-x));

    for n = 2:(N+1)
        % Level 1 prediction
        mu1hat(n,:) = ux(mu2(n-1,:));                       % Eq 24
    
        % Identify valid blocks
        valid = ~isnan(y(n,:));
    
        % Default: carry forward all blocks
        mu2(n,:)    = mu2(n-1,:);
        sigma2(n,:) = sigma2(n-1,:);
        mu3(n,:)    = mu3(n-1,:);
        sigma3(n,:) = sigma3(n-1,:);
    
        expmu3 = exp(omega + kappa .* mu3(n-1,:));     
    
        % Level 1 prediction error (valid blocks only)
        delta1    = y(n,valid) - mu1hat(n,valid);           % Eq 25
        sigma1hat = mu1hat(n,valid) .* (1 - mu1hat(n,valid)); % Eq 26
    
        % Level 2 update
        sigma2hat = sigma2(n-1,valid) + expmu3(valid);      % Eq 27
        pi2       = (sigma2hat.^-1) + sigma1hat;            % Eq 28
        LR        = pi2.^-1;
    
        mu2(n,valid)    = mu2(n-1,valid) + LR .* delta1;    % Eq 23
        sigma2(n,valid) = LR;                               % Eq 22
    
        % Level 3 update
        pihat3 = (sigma3(n-1,valid) + nu).^-1;              % Eq 31
        w2     = expmu3(valid) ./ (expmu3(valid) + sigma2(n-1,valid)); % Eq 32
        r2     = (expmu3(valid) - sigma2(n-1,valid)) ./ ...
                 (expmu3(valid) + sigma2(n-1,valid));       % Eq 33
    
        delta2 = (sigma2(n,valid) + ...
                  (mu2(n,valid) - mu2(n-1,valid)).^2) ./ ...
                  (sigma2(n-1,valid) + expmu3(valid)) - 1;  % Eq 34
    
        pi3 = pihat3 + (kappa^2/2) .* w2 .* (w2 + r2 .* delta2); % Eq 29
        if any(pi3 <= 0)
            bad_traj = true;
        end
    
        sigma3(n,valid) = pi3.^-1;
        mu3(n,valid)    = mu3(n-1,valid) + ...
                          (kappa/2) .* sigma3(n,valid) .* w2 .* delta2; % Eq 30
    end

    vars = [mu2(:); mu3(:); sigma2(:); sigma3(:)];
    % if any(isnan(vars(:)))
    %     bad_traj = true;
    % end

    predictions = mu1hat(2:end, :);


    % --- blockwise GLM (same lr analysis as your distr-HMM) ---
    % Relate update to delta with block-specific intercepts.
    % This produces a single slope per block (compact summary).
    V = predictions;
    delta  = outcomes-V;  % (o - V) at each t (but Vmat stored at t)
    delta(end,:) = [];         % align with update (t -> t+1)
    update = V(2:end,:) - V(1:end-1,:);

    y = [];
    x = [];
    s_const = [];
    
    
    for i = 1:size(delta,2)
        y = [y; update(:,i)];
        x = blkdiag(x, delta(:,i));
        s_const = blkdiag(s_const, ones(size(delta,1),1));
    end
    trial_lr = update./delta;
    
    b_glm = glmfit([x s_const], y, 'normal', 'constant', 'off');

    out = struct();

    out.glm_lr = b_glm(1:size(delta,2),:);           % slope per block (update ~ delta)
    out.block_effect = b_glm(size(delta,2)+1:end,:); % intercept per block

    out.bad_traj = bad_traj;
    out.mu1hat = mu1hat(2:end,:);   % N x B predicted outcome prob
    out.mu2    = mu2;
    out.mu3    = mu3;   % N x B volatility state
    out.sigma2 = sigma2;
    out.sigma3 = sigma3;
    out.LR2    = sigma2(2:end,:);   % N x B learning rate
    out.trial_lr = trial_lr;
    out.vol = exp(omega + kappa .* mu3); % expmu3  
end

