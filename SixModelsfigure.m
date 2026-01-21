% Define file and sheet
close all
clear all
clc
filename = matlab.desktop.editor.getActiveFilename;
[~, filename, ~] = fileparts(filename);

figure

%% Moon Model
load('Moon_replicates_50000_app.mat')
d=20;
labels = arrayfun(@(k) sprintf('X_{%d}', k), 1:d, 'UniformOutput', false);

subplot(3,1,1)
b = bar(AANN');  b(1).FaceColor = [0 0 0]; b(2).FaceColor = [0 0 1]; b(3).FaceColor = [1 0 0]; b(4).FaceColor=[0 1 0];
title('Moon Model','FontSize',30,'Interpreter','latex')
ticks=[1:20];
%xticks(ticks)
%xticklabels(labels(sort(bb(1:7))))
%%xtickangle(45)
%set(gca,'TickLabelInterpreter','none')
%xtickangle(90); % Set orientation to vertical
set(gca,'FontSize',26)
set(gcf,'WindowState','maximized')


%% Model99
load('Model99_replicates.mat')

% Select desired indices
idx = [1 2 3 50 51 52 53 82:83 86:88 92:93 97:99];
d = length(idx);

% Generate labels
labels = arrayfun(@(k) sprintf('%d', k), idx, 'UniformOutput', false);
labels{3} = '...';   % Replace label for X_3
labels{7} = '...';   % Replace label for X_53

% Extract subset of data
AANN_sel_99=AANN(:, idx);

% Plotting
subplot(3,1,2)
b = bar(AANN_sel_99');  
b(1).FaceColor = [0 0 0]; 
b(2).FaceColor = [0 0 1]; 
b(3).FaceColor = [1 0 0]; 
b(4).FaceColor = [0 1 0];
title('Model 99','FontSize',30,'Interpreter','latex')
xticks(1:d)
xticklabels(labels)
xtickangle(0)
legend('Ground Truth','$\nu_j$', '$\nu_j^{GCMR}$', '$\nu_j^{GKnock}$', ...
    'FontSize',28,'Interpreter','latex','Location','Northwest')
set(gca,'FontSize',26)
set(gcf,'WindowState','maximized')


%% Model500

filename500 = 'Model500_0550000_0.5.xlsx';
sheet = 'NeuralNetwork';
rangeGT = 'A2:SF2';
rangeNN = 'A3:SF5';
groundtuth=readmatrix(filename500,'FileType','spreadsheet','Sheet',sheet,'Range',rangeGT);
AANN = readmatrix(filename500,'FileType','spreadsheet','Sheet',sheet,'Range',rangeNN);

% Select desired indices
idx = [2:4 19:21 31:32 255:257 271:273 278 281:282];
k = length(idx);
d=k;
% Generate labels
labels = arrayfun(@(k) sprintf('%d', k), idx, 'UniformOutput', false);
for i=1:2
labels{6+i} = '...'; 
labels{12} = '...';
labels{find(idx==278)}='...';
labels{16+i} = '...';
end
% Replace label for X_3

   % Replace label for X_53

% Extract subset of data
AANN_sel_500= [groundtuth(idx); AANN(:, idx)];

%%

subplot(3,1,3)
b = bar(AANN_sel_500');  
b(1).FaceColor = [0 0 0]; 
b(2).FaceColor = [0 0 1]; 
b(3).FaceColor = [1 0 0]; 
b(4).FaceColor = [0 1 0];
title('Model 500','FontSize',30,'Interpreter','latex')
xticks(1:d)
xticklabels(labels)
xtickangle(0)
set(gca,'FontSize',26)
set(gcf,'WindowState','maximized')


%subplot(3,1,3)
%b = bar(AARF_sel');  
%b(1).FaceColor = [0 0 0]; 
%b(2).FaceColor = [0 0 1]; 
%b(3).FaceColor = [1 0 0]; 
%b(4).FaceColor = [0 1 0];
%title('Random Forest','FontSize',30,'Interpreter','latex')
%xticks(1:d)
%xticklabels(labels)
%xtickangle(0)
%set(gca,'FontSize',22)
%set(gcf,'WindowState','maximized')
%
saveas(gcf,['../FiguresUsedinthePaper/',filename],'fig')
saveas(gcf,['../FiguresUsedinthePaper/',filename],'eps')
saveas(gcf,['../FiguresUsedinthePaper/',filename],'jpg')
saveas(gcf,['../FiguresUsedinthePaper/',filename],'png')
