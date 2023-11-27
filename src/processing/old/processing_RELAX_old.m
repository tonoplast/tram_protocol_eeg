close all; clear;

% Path (repo and cap location)
userprofilePath = getenv('USERPROFILE');
repoPath = [userprofilePath, filesep 'GitHub' filesep 'tram_protocol_eeg'];
configPath = [repoPath, filesep 'src' filesep 'config'];

% config
cfg = jsondecode(fileread([configPath filesep 'config_matlab.json'])).processing;

% Example
% which_drive = 'D:';
% set_srate = 256; % downsampling (if not already done)
% file_extension = '.set'; % file extention to run
% channels_to_remove = ['FP1','FP2']; % an empty list with quotes ['']
% rereference_without_these_electrodes = [''];

% paths
eeglab_path = [cfg.which_drive, filesep 'tram_protocol_eeg' filesep 'toolbox' filesep 'eeglab2022.1'];
inPath = [cfg.which_drive, filesep 'EEG_data_collected_today' filesep 'processed_EEG' filesep 'RELAXProcessed' filesep 'Cleaned_Data'];
outPath = [inPath, filesep 'ready_for_TGC'];

if not(isfolder(outPath))
    mkdir(outPath)
end





% loading eeglab
cd(eeglab_path);
[ALLEEG EEG CURRENTSET ALLCOM] = eeglab;

% Get a list of all the files and folders in the folder
files = dir(inPath);

% Loop through each file/folder
for i = 1:numel(files)
    % Skip over the '.' and '..' entries, which refer to the current and parent directories
    if strcmp(files(i).name, '.') || strcmp(files(i).name, '..')
        continue;
    end
    
    % If the current file/folder is a directory, skip them
    if files(i).isdir
        % Display the name of the folder
        disp(['Found folder: ' files(i).name '. Will ignore.']);
    elseif contains(files(i).name, cfg.file_extension)
        file_to_process = fullfile(inPath, files(i).name);
        disp(['Processing: ' file_to_process])

        [pathstr, filename, ext] = fileparts(file_to_process);

%         IDs = strsplit(filename, '_');
%         ID = string(IDs(1));

        % loading data (keystrokes off)
        EEG = pop_loadset(file_to_process);
        EEG = eeg_checkset( EEG );
        
        % removing events
        EEG.event = [];
        EEG.urevent = [];
	
		% interpolate missing channels (RELAX removes them)
		EEG = pop_interp(EEG, EEG.allchan, 'spherical'); 
        [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 1,'gui','off'); 
        EEG = eeg_checkset( EEG );
        
        % in case want to remove specific channels
        remove_channel_index = find(ismember({EEG.chanlocs.labels}, cfg.channels_to_remove));
        EEG = pop_select(EEG, 'nochannel', remove_channel_index);
        
		% average re-referencing
        rereference_without_these_electrodes = find(ismember({EEG.chanlocs.labels}, cfg.rereference_without_these_electrodes));
		EEG = pop_reref(EEG, [],'exclude', rereference_without_these_electrodes);
		
        % downsampling
        EEG = pop_resample(EEG, cfg.set_srate); %% down-sampling should reduce time it takes.
        [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 1,'gui','off'); 


        pop_saveset(EEG, 'filename', [filename '_rereferenced.set'],'filepath', outPath);
        
        STUDY = []; CURRENTSTUDY = 0; ALLEEG = []; EEG=[]; CURRENTSET=[];

    else
        disp(['Not processing: ' files(i).name])
    end

end

