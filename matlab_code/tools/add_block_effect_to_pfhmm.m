function add_block_effect_to_pfhmm(experiment, postfix, rho)

    % Determine the current folder (location of this file) and set the working directory
    currentFolder = fileparts(mfilename('fullpath'));
    cd(currentFolder);
    fdir = fullfile('..', sprintf('experiment_%s', experiment));
    fname = fullfile(fdir, sprintf('pfhmm_%sfit_%s%s.mat', rho, experiment, postfix));

    f = load(fname);
    obs_dynamics = f.obs_dynamics;
    [data, ~, ~] = get_data(experiment);

    Nsub = numel(obs_dynamics);

    for n = 1:Nsub
        dyn = obs_dynamics{n};

        val = dyn.val;          % T x B
        outcomes = data{n}.outcome;
        choices  = data{n}.choice;
        outcomes(isnan(choices)) = NaN;   % match PFHMM masking


        [T, B] = size(val);

        lr = nan(1,B);
        block_effect = nan(1,B);

        for b = 1:B
            delta  = outcomes(1:end-1,b) - val(1:end-1,b);
            update = val(2:end,b) - val(1:end-1,b);

            valid = ~isnan(delta);
            delta  = delta(valid);
            update = update(valid);

            if numel(delta) < 5
                continue
            end

            X = [delta, ones(numel(delta),1)];
            beta = glmfit(X, update, 'normal', 'constant', 'off');

            lr(b) = beta(1);
            block_effect(b) = beta(2);
        end

        obs_dynamics{n}.block_effect = block_effect;
        obs_dynamics{n}.lr_glm       = lr;
    end

    save(fname, '-append', 'obs_dynamics');
end
