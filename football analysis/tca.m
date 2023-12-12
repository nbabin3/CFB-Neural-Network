load stats_2016_2023
load gameInputs_2016_2023_full
load statsKeys
load teams

teams = cellstr(teams);
statsKeys = cellstr(statsKeys);
statsKeys(152:157) = [];

stats_2016_2023(:, 152:157, :) = [];
gameInputs(:, [153:158, 312:317]) = [];

invertedStats = [2:9, 13, 14, 26, 28, 29, 40, 42, 59, 82];
for i = invertedStats
    stats_2016_2023(:, i, :) = -stats_2016_2023(:, i, :);
end

stats = tensor(stats_2016_2023);
statsD = size(stats);

numStats = statsD(2);
numSeasons = 7;

R = 3;

model = ncp(stats, R, 'verbose', 0);
err = norm(minus(full(model), full(stats)))/norm(stats);

model_teams = model.U{1};  % neural factor
model_stats = model.U{2};    % time factor
model_seasons = model.U{3};   % trial factor

[~, statsSortOrder] = sort(model_stats(:, 1), 'descend');

seasons = {'2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023'};

stats_cat = categorical(statsKeys(statsSortOrder));
teams_cat = categorical(teams);
seasons_cat = categorical(seasons);

% sort first component stats factor descending for visibility
stats_cat = reordercats(stats_cat, statsKeys(statsSortOrder));

bord = .05;
% ylim_teams = [min(model_teams(:)) max(model_teams(:))];
% ylim_stats = [min(model_stats(:)) max(model_stats(:))];
% ylim_seasons = [min(model_seasons(:))-bord max(model_seasons(:))+bord];

stats_c = zeros(length(model_stats(:,1)),R);

% plot factors for each latent component
for i = 1:R
    modelStats = model_stats(:, i);

    subplot(R, 3, 3*i - 2)
    bar(stats_cat, modelStats(statsSortOrder), 'b')
    set(gca, 'XTickLabel', {})

%     ylim(ylim_stats)

    if i == 1
        title("Stats Factor")
    end
    if i == R
        xlabel("Stats")
    end

    inc = .15;
%     yticks(floor(ylim_stats(1)/inc)*inc:inc:ceil(ylim_stats(2)/inc)*inc)

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    subplot(R, 3, 3*i - 1)

    bar(teams_cat, model_teams(:, i))
    hold on
    set(gca, 'XTickLabel', {})
%     ylim(ylim_teams)
    if i == 1
        title("Team Factor")
    end
    if i == R
        xlabel("Teams")
    end

    inc = .15;
%     yticks(floor(ylim_teams(1)/inc)*inc:inc:ceil(ylim_teams(2)/inc)*inc)

    stats_c(:, i) = model_stats(:, i);


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    subplot(R, 3, 3*i)
    % optionally plot trial factor color-coded by number of bar
    
    bar(seasons_cat, model_seasons(:, i))
    hold on
    inc = .15;
%     ylim(ylim_seasons)
%     yticks(floor(ylim_seasons(1)/inc)*inc:inc:ceil(ylim_seasons(2)/inc)*inc)
    if i == R
        xlabel("Season")
    end
%             colormap("cool")
%             colorbar
    
%     xinc_seasons = 250;
%     xticks(floor(min(flagTimes)/xinc_seasons)*xinc_seasons:xinc_seasons:ceil(max(flagTimes)/xinc_seasons)*xinc_seasons)


    if i == 1
        title("Season Factor")
    end
end
set(gcf, 'Name', strcat("CFB Stats Dimensionality Reduction, fit: ", string(1 - err)))
fontsize(gcf, 28, "points")
figure

statsCC = zeros(numStats);
winCC = zeros(numStats, 1);

for s1 = 1:numStats
    for s2 = 1:numStats
        statsCC(s1, s2) = corr2(squeeze(stats_2016_2023(:, s1, :)), squeeze(stats_2016_2023(:, s2, :)));
    end
%     winCC(s1) = corr2(squeeze(stats_2015_2022(:, s1, :)), winPct);
end
statsCC2 =statsCC.^2;
% winCC2 = winCC.^2;

redundancy = sum(statsCC2, 1)/numStats;
[redundSort, redundSortOrd] = sort(redundancy, "ascend");
statsKeys_redundSort = categorical(statsKeys(redundSortOrd));
statsKeys_redundSort = reordercats(statsKeys_redundSort, string(statsKeys(redundSortOrd)));

% [winSort, winSortOrd] = sort(winCC2, "descend");
% statsKeys_winSort = categorical(statsKeys(winSortOrd));
% statsKeys_winSort = reordercats(statsKeys_winSort, string(statsKeys(winSortOrd)));

heatmap(statsCC2, 'XData', statsKeys, 'YData', statsKeys)
colormap("parula")
figure

save stats_2016_2023_clean.mat stats_2016_2023
save gameInputs_2016_2023_clean.mat gameInputs

% bar(statsKeys_redundSort, redundSort)
% figure
% 
% bar(statsKeys_winSort, winSort)
% figure

% statsToKeep = [];
% statsToRemove = [];
% cc2Thresh = .9;

% for r = 1:length(statsKeys)
%     for s = r+1:length(statsKeys)
%         if any(strcmpi(statsToRemove, statsKeys(r))) || any(strcmpi(statsToRemove, statsKeys(s)))
%             continue
%         end
% 
%         if statsCC2(r, s) > cc2Thresh
%             if winCC2(r) > winCC2(s)
%                 statsToRemove = [statsToRemove; s];
%                 statsToKeep = [statsToKeep, r];
%             else
%                 statsToRemove = [statsToRemove; r];
%                 statsToKeep = [statsToKeep, s];
%             end
%         end
%     end
% end

% statsToRemove = unique(statsToRemove);

% numStatsCond = numStats - length(statsToRemove);
% statsCond = zeros(length(teams), numStatsCond, numSeasons);
% 
% statsKeysCond = cell(numStatsCond, 1);
% 
% x=0;
% for r = 1:length(statsKeys)
%     if ~ismember(r, statsToRemove)
%         x = x+1;
%         statsCond(:, x, :) = stats(:, r, :);
%         statsKeysCond(x) = statsKeys(r);
%     end
% end
% 
% statsCondCC = zeros(numStatsCond);
% 
% for s1 = 1:numStatsCond
%     for s2 = 1:numStatsCond
%         statsCondCC(s1, s2) = corr2(squeeze(statsCond(:, s1, :)), squeeze(statsCond(:, s2, :)));
%     end
% end
% 
% statsCondCC2 =statsCondCC.^2;
% 
% heatmap(statsCondCC2, 'XData', statsKeysCond, 'YData', statsKeysCond)
% colormap("parula")
% figure
% 
% % statsCond = tensor(statsCond);
% % writematrix("statsKeysCond.csv", statsKeysCond)
% 
% save stats_2015_2019_condensed.mat statsCond

