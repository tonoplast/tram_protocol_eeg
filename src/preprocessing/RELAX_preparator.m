close all; clear;

% config
% Path (repo and cap location)
userprofilePath = getenv('USERPROFILE');
repoPath = [userprofilePath, filesep 'GitHub' filesep 'tram_protocol_eeg'];
configPath = [repoPath, filesep 'src' filesep 'config'];

% config (preparation)
cfg = jsondecode(fileread([configPath filesep 'config_matlab.json'])).preparation;

eeglab_path = [repoPath, filesep 'src' filesep 'toolbox' filesep cfg.eeglab_version];

cd(eeglab_path);
[ALLEEG EEG CURRENTSET ALLCOM] = eeglab;


% Path (loading / saving)
inPath = [cfg.data_drive, filesep cfg.starting_folder];
outPath = [inPath, filesep 'processed_EEG'];

if not(isfolder(inPath))
    mkdir(inPath)
end

if not(isfolder(outPath))
    mkdir(outPath)
end


% Get a list of all the files and folders in the folder
files = dir(inPath);

if numel(files) == 0
    error('There is no file in the "EEG_data_collected_today" folder! Please make sure the raw files are in there.')
end

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

        % loading data
        if contains(ext,  '.cnt')
            EEG = pop_loadcnt(file_to_process, 'dataformat', 'auto' ,'memmapfile', '');
        elseif contains(ext,  '.bdf')
            EEG = pop_biosig(file_to_process, 'ref', 48); %% 48 is Cz (but will become 47 after removing elecs below)
        
            % find middle point and extract 6 min worth of data 
            % (180 sec each side)
            start_pnt = floor(EEG.xmax/2 - 180) * EEG.srate;
            end_pnt = ceil(EEG.xmax/2 + 180) * EEG.srate;

            % Create a start_event structure with the same fields as EEG.event
            start_event = struct('type', 'start', 'edftype', 888, 'latency', start_pnt, 'urevent', 'start');
            end_event = struct('type', 'end', 'edftype', 999, 'latency', end_pnt, 'urevent', 'end');
            
            % Check if EEG.event is empty
            if isempty(EEG.event)
                EEG.event = [start_event, end_event];
            else
                % Check if 'edftype' field does not exist in EEG.event
                if ~isfield(EEG.event, 'edftype')
                    % If it doesn't exist, add 'edftype' to all events with a default value (e.g., 0)
                    [EEG.event.edftype] = deal(0);
                end

                % Concatenate start_event and end_event with EEG.event
                EEG.event = [EEG.event, start_event, end_event];
            end
            
            % Concatenate start_event with EEG.event
            EEG.event = [EEG.event start_event];
            EEG.event = [EEG.event end_event];
    
            EEG = pop_rmdat( EEG, {'start'},[0 360] ,0);
            [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 1,'gui','off'); 
            EEG = eeg_checkset( EEG );
           
        end

        EEG = eeg_checkset( EEG );

        EEG.NoCh = {'GSR1'; 'GSR2'; 'Erg1'; 'Erg2'; 'Resp'; 'Plet'; 'Temp'; 'Iz'};
        EEG = pop_select(EEG,'nochannel',EEG.NoCh);  
        
        % removing events
        EEG.event = [];
        EEG.urevent = [];

        [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 1,'gui','off'); 
        EEG = eeg_checkset( EEG );

        EEG = pop_chanedit(EEG, 'lookup', [configPath filesep cfg.caploc]);
        [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 1,'gui','off'); 
        EEG = eeg_checkset( EEG );
        
        
        % keeping these eeg channels
        EEG = pop_select(EEG, 'channel', cfg.channels_to_keep);
        EEG.allchan=EEG.chanlocs;
        
        [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 1,'gui','off'); 
        EEG = eeg_checkset( EEG );

        % downsampling
        if EEG.srate > cfg.set_srate
            EEG = pop_resample(EEG, cfg.set_srate); %% down-sampling should reduce time it takes.
            [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 1,'gui','off'); 
        end

        pop_saveset(EEG, 'filename', [filename '_downsampled.set'],'filepath', outPath);
        
        STUDY = []; CURRENTSTUDY = 0; ALLEEG = []; EEG=[]; CURRENTSET=[];

    else
        disp(['Not processing: ' files(i).name])
    end

end

