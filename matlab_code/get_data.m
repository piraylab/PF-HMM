function [data, metadata] = get_data(experiment)
% GET_DATA Load experimental trial data and metadata for a given experiment.
%
%   [data, metadata] = get_data(experiment)
%
%   Inputs:
%       experiment   - A string (or char array) identifying the experiment.
%                      If not provided, defaults to 'sealion'.
%
%   Outputs:
%       data         - A cell array where each cell contains a struct with
%                      trial-specific data (choice, outcome, randomization order,
%                      response time, and workerId).
%       metadata     - A struct that contains metadata about the experiment,
%                      including additional hidden state information.
%
%   The function loads a .mat file based on the experiment identifier,
%   processes the trial data by replacing invalid choice values (-1) with NaN,
%   and appends hidden state metadata from a separate .mat file.

    % If no experiment is specified, default to 'sealion'
    if nargin < 1
        experiment = 'sealion';
    end
    
    % Print out the experiment being processed for the user's information
    fprintf('Experiment: %s\n', experiment);
    
    % =========================================================================
    % Construct the filename and path for the experiment data file
    % =========================================================================
    
    % Get the full path of the current file (the location of this function)
    currentFolder = fileparts(mfilename('fullpath'));
    
    % Change directory to the current folder to ensure relative paths work correctly
    cd(currentFolder);
    
    % Define the folder where the .mat files are stored.
    % This example assumes the data is in the parent folder under 'mat_data'
    fdir = fullfile('..', 'mat_data');
    
    % Build the filename using the experiment identifier (data_experiment.mat)
    fname = fullfile(fdir, sprintf('data_%s.mat', experiment));
    
    % Load the .mat file; it is expected to contain a struct with at least the field 'trials_data'
    f = load(fname);
    
    % =========================================================================
    % Process trial data from the loaded file
    % =========================================================================
    
    % Determine the number of trials in the trial data cell array
    N = length(f.trials_data);
    
    % Extract the cell array of trial data
    trials_data = f.trials_data;
    
    % Initialize the output cell array for processed trial data
    data = cell(N, 1);
    
    % Loop over each trial and process its data
    for i = 1:N
        % Check if the 'choice' field in the trial contains -1, which denotes an invalid choice
        if any(trials_data{i}.choice(:) == -1)
            % Replace all -1 values with NaN (Not a Number) for proper handling of missing data
            trials_data{i}.choice(trials_data{i}.choice == -1) = NaN;
        end
        
        % Create a new struct for the current trial in the output cell array
        % and assign the relevant fields from the original data
        data{i}.choice        = trials_data{i}.choice;
        data{i}.outcome       = trials_data{i}.outcome;
        data{i}.rand_order    = trials_data{i}.randomization_order;
        data{i}.response_time = trials_data{i}.response_time;
        data{i}.workerId      = trials_data{i}.workerId;
    end
    
    % =========================================================================
    % Process metadata and append hidden state information
    % =========================================================================
    
    % Extract the metadata from the loaded file (assumed to be stored as 'meta_data')
    metadata = f.meta_data;
    
    % Define the filename for the additional hidden state time series data
    % The file is expected to be in the same folder as the main data file
    timeseries_fname = fullfile(fdir, 'hidden_state.mat');
    
    % Load the hidden state file which contains time series information
    timeseries_f = load(timeseries_fname);
    
    % Append the hidden state data into the metadata struct under a new field 'hidden_state'
    metadata.hidden_state = timeseries_f.timeseries.hidden_state;
end

