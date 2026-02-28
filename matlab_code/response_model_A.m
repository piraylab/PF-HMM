function out = response_model_A(p1_pred, choice01, params)
% RESPONSE_MODEL  Binary choice response with optional biases and perseveration
%
% Inputs
%   p1_pred  : N x B predicted P(choice==1) from cognitive model
%   choice01 : N x B observed choices (0/1), NaN allowed
%
% Optional fields in params:
%   params.bv   : main-effect bias (factor 1)
%   params.be   : main-effect bias (factor 2)
%   params.bi   : interaction bias
%   params.rho  : perseveration strength (logit bias from previous choice)
%   params.beta : inverse temperature (evidence gain; default = 1)
%
% Outputs
%   out.p1        : N x B final P(choice==1)
%   out.p_choice  : N x B probability of observed choice
%   out.loglik_n  : vector of trial-wise log-likelihoods
%   out.loglik    : scalar total log-likelihood

    addpath('tools');
    % defaults
    if nargin < 3, params = struct(); end

    bv   = getfield_def(params, 'bv',   0);
    be   = getfield_def(params, 'be',   0);
    bi   = getfield_def(params, 'bi',   0);
    rho  = getfield_def(params, 'rho',  0);
    beta = getfield_def(params, 'beta', 1);

    p = p1_pred;
    [T, B] = size(p);

    % logit of cognitive prediction 
    d = log(p ./ (1 - p));                % N x B

    % scale evidence
    d = beta * d;

    % condition-specific biases (2x2 design)
    % Block order assumed:
    %   1: +v +e
    %   2: +v -e
    %   3: -v +e
    %   4: -v -e
    if B == 4
        bb = [ bv  bv -bv -bv ] + ...
             [ be -be  be -be ] + ...
             [ bi -bi -bi  bi ];
        d = d + bb;                       % broadcast across trials
    end

    % perseveration (optional)
    if rho ~= 0
        prev_choice = [nan(1,B); choice01(1:end-1,:)];
        d(prev_choice == 1) = d(prev_choice == 1) + rho;
        d(prev_choice == 0) = d(prev_choice == 0) - rho;
    end

    % back to probability
    p1 = 1 ./ (1 + exp(-d));

    % likelihood
    p_choice = p1 .* choice01 + (1 - p1) .* (1 - choice01);
    valid = ~isnan(choice01);
    loglik_n = log(p_choice(valid) + eps);
    loglik = sum(loglik_n);

    % outputs
    out.p1       = p1;
    out.p_choice = p_choice;
    out.loglik_n = loglik_n;
    out.loglik   = loglik;
end


