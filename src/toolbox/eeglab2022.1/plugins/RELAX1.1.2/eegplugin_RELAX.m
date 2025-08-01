%% RELAX EEG CLEANING PIPELINE, Copyright (C) (2022) Neil Bailey

%     This program is free software: you can redistribute it and/or modify
%     it under the terms of the GNU General Public License as published by
%     the Free Software Foundation, either version 3 of the License, or
%     any later version.
% 
%     This program is distributed in the hope that it will be useful,
%     but WITHOUT ANY WARRANTY; without even the implied warranty of
%     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%     GNU General Public License for more details.
% 
%     You should have received a copy of the GNU General Public License
%     along with this program.  If not, see https://www.gnu.org/licenses/.

%% eegplugin_RELAX:
% Add RELAX to the EEGLAB gui:
function vers = eegplugin_RELAX(fig, try_strings, catch_strings)

    vers = 'RELAX 1.1.2';

    if ispc      % windows
            wfactor1 = 1.20;
            wfactor2 = 1.21;
    elseif ismac % Mac OSX
            wfactor1 = 1.45;
            wfactor2 = 1.46;
    else
            wfactor1 = 1.30;
            wfactor2 = 1.31;
    end
    posmainfig = get(gcf,'Position');
    hframe     = findobj('parent', gcf,'tag','Frame1');
    posframe   = get(hframe,'position');
    set(gcf,'position', [posmainfig(1:2) posmainfig(3)*wfactor1 posmainfig(4)]);
    set(hframe,'position', [posframe(1:2) posframe(3)*wfactor2 posframe(4)]);

    menuRELAX = findobj(fig,'tag','EEGLAB');   % At EEGLAB Main Menu

    submenu = uimenu( menuRELAX,'Label','RELAX','separator','on','tag','RELAX','userdata','startup:on;continuous:on;epoch:on;study:on;erpset:on');
    
    % menu callbacks
        % --------------
    comProcessData = [try_strings.no_check...
        '[RELAX_cfg, FileNumber, CleanedMetrics, RawMetrics, RELAXProcessingRoundOneAllParticipants, RELAXProcessingRoundTwoAllParticipants, RELAXProcessing_wICA_AllParticipants, RELAXProcessing_ICA_AllParticipants, RELAXProcessingRoundThreeAllParticipants, RELAX_issues_to_check, RELAXProcessingExtremeRejectionsAllParticipants] = pop_RELAX();'...
        catch_strings.add_to_hist];
     % create menus
        % -------------------------   
    uimenu( submenu, 'Label', 'Preprocess EEG Data'  , 'CallBack', comProcessData);
    
    % menu callbacks
        % --------------
    comProcessData_beta = [try_strings.no_check...
        '[RELAX_cfg, FileNumber, CleanedMetrics, RawMetrics, RELAXProcessingRoundOneAllParticipants, RELAXProcessingRoundTwoAllParticipants, RELAXProcessing_wICA_AllParticipants, RELAXProcessing_ICA_AllParticipants, RELAXProcessingRoundThreeAllParticipants, RELAX_issues_to_check, RELAXProcessingExtremeRejectionsAllParticipants] = pop_RELAX_beta();'...
        catch_strings.add_to_hist];
     % create menus
        % -------------------------   
    uimenu( submenu, 'Label', 'Preprocess EEG Data (beta version)'  , 'CallBack', comProcessData_beta);
    
    % menu callbacks
        % --------------
    comEpochData = [try_strings.no_check...
        '[OutlierParticipantsToManuallyCheck,EpochRejections,RELAX_epoching_cfg] = pop_RELAX_epoch_the_clean_data();'...
        catch_strings.add_to_hist];
     % create menus
        % -------------------------   
    uimenu( submenu, 'Label', 'Epoch Data, Reject Bad Epochs and BL Correct EEG Data'  , 'CallBack', comEpochData);
    
    % menu callbacks
        % --------------
    comHelpMenu = [try_strings.no_check...
        '[RELAX_wiki_website] = pop_RELAX_help();'...
        catch_strings.add_to_hist];
     % create menus
        % -------------------------   
    uimenu( submenu, 'Label', 'Help'  , 'CallBack', comHelpMenu);
    
    % menu callbacks
        % --------------
    comCitationDetails = [try_strings.no_check...
        '[RELAX_citation] = pop_RELAX_citation();'...
        catch_strings.add_to_hist];
     % create menus
        % -------------------------   
    uimenu( submenu, 'Label', 'Citing RELAX'  , 'CallBack', comCitationDetails);
    
end
