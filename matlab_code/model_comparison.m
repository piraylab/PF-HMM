function model_comparison()
    % This function performs a model comparison analysis using Bayesian
    % Model Selection (BMS) for two models whose results are stored in .mat files.
    % It loads the log evidence for each participant from two different model fits,
    % computes the model frequencies and posterior exceedance probabilities (pxp)
    % using the cbm_spm_BMS function, and then saves the results.

    % Specify the directory where the .mat files are located.
    % 'fullfile' builds a full file path. Here, '..' means one level up,
    % and 'mat_data' is the folder containing the data.
    fdir = fullfile('..', 'mat_data');

    % Prepare filenames for the two models.
    % Model 1: 'hmm_fit.mat'
    % Model 2: 'hmm_beta_fit.mat'
    fnames{1} = fullfile(fdir, sprintf('hmm_fit.mat')); 
    fnames{2} = fullfile(fdir, sprintf('hmm_beta_fit.mat')); 

    % Load the first model file to determine the number of participants.
    % The file is expected to contain a variable 'cbm', which is a cell array,
    % with one entry per participant.
    f = load(fnames{1}); 
    num_participants = length(f.cbm);

    % Preallocate an array to store the log evidence values.
    % log_evidence will have one row per participant and one column per model.
    log_evidence = nan(num_participants, 2);

    % Loop over each model file (i.e., for two models).
    for i = 1:2
        % Load the i-th model results file.
        f = load(fnames{i});
        % Extract the 'cbm' cell array from the loaded file.
        cbm = f.cbm;
        % For each participant, extract the log evidence value.
        % Assume each participant's result is stored in a structure under
        % the field 'output.log_evidence'.
        for n = 1:length(cbm)
            log_evidence(n, i) = cbm{n}.output.log_evidence;
        end
    end

    % Add the 'cbm' folder to the MATLAB search path.
    % This is necessary because the function 'cbm_spm_BMS' resides in that folder.
    addpath('cbm');

    % Run Bayesian Model Selection (BMS) using the log evidence matrix.
    % The function cbm_spm_BMS takes the log evidence and outputs:
    %   1. (unused here, denoted by ~),
    %   2. model_frequency: the frequency of each model among participants,
    %   3. (unused here, denoted by ~),
    %   4. pxp: the posterior exceedance probability for each model.
    [~, model_frequency, ~, pxp] = cbm_spm_BMS(log_evidence);

    % Store the BMS results in a structure for later use or inspection.
    bms_results = struct('model_frequency', model_frequency, ...
                         'pxp', pxp, ...
                         'num_participants', num_participants);

    % Save the results to a .mat file.
    % The file will be saved in the 'mat_data' folder with the same name as the function.
    % mfilename returns the name of the current function ('model_comparison').
    fname = fullfile(fdir, sprintf('%s.mat', mfilename));
    save(fname, "bms_results");

end