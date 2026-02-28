function S = summarize_PH_alpha(signals)
%SUMMARIZE_PH_ALPHA Robustly summarize Pearce-Hall alpha across subjects.
%
% Handles subjects with different trial counts (T) and/or blocks (B)
% by padding with NaN to the max size.

    N = numel(signals);

    % --- find max sizes across subjects ---
    Ts = nan(N,1);
    Bs = nan(N,1);
    for n = 1:N
        a = signals{n}.alpha;
        Ts(n) = size(a,1);
        Bs(n) = size(a,2);
    end
    Tmax = max(Ts);
    Bmax = max(Bs);

    % --- preallocate padded container ---
    alpha_sub = nan(N, Tmax, Bmax);
    alpha_blockmean = nan(N, Bmax);

    % --- fill subject-by-subject ---
    for n = 1:N
        a = signals{n}.alpha;   % size can vary: [Tn x Bn]
        Tn = size(a,1);
        Bn = size(a,2);

        % copy available portion
        alpha_sub(n, 1:Tn, 1:Bn) = a;

        % block mean over available trials
        alpha_blockmean(n, 1:Bn) = mean(a, 1, 'omitnan');
    end

    % --- group summaries ---
    S = struct();
    S.alpha_sub = alpha_sub;                 % [N x Tmax x Bmax] padded with NaN
    S.alpha_blockmean = alpha_blockmean;     % [N x Bmax]

    S.mean_alpha_per_block = mean(alpha_blockmean, 1, 'omitnan');
    S.sem_alpha_per_block  = std(alpha_blockmean, 0, 1, 'omitnan') ./ ...
                             sqrt(sum(~isnan(alpha_blockmean),1));

    fprintf("Mean alpha per block: %s\n", mat2str(S.mean_alpha_per_block, 4));
end
