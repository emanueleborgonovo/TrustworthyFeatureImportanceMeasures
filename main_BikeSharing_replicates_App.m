close all;clearvars;clc

%Alreadytrained=input('Already Trained?; 1 if true, 0 if false ');
Rep=input('Number of Replicates ');


Data = importfile_BikeSharing('hour.csv');
filename = 'BikeSharing'
%removecolumns = {};
%Data(:,removecolumns) = [];
summary(Data)

% Output variable
targetVar = 'cnt';
targetVar2='casual';
targetVar3='registered';

% Separate predictors and target
y = Data.(targetVar);  % Extract target variable (as vector)
x = removevars(Data, targetVar);  % Remove 'cnt' column from table
x = removevars(x, targetVar2);
x = removevars(x, targetVar3);

labels=x.Properties.VariableNames;
x = table2array(x);
[n,d]=size(x);

%writetable(Data, 'bikesharing_data.csv');
%%
filename = matlab.desktop.editor.getActiveFilename;
[~, filename, ~] = fileparts(filename);
netname=strcat('trainednet',filename);
netnamemat=strcat(netname,'.mat')
%% Calculate wassersi -  not run now
VY=var(y);
[W]=wassersi(x,y,25);
Wass2=W.W22/(2*var(y));
figure
bar(Wass2)
title('Statistical Association ($\iota^{W_2^2}$)','Interpreter','Latex')
xticklabels(labels)
ylabel('$\hat{\iota}^{W_2^2}$','Interpreter','Latex')
set(gca,'TickLabelInterpreter','none')
set(gca,'FontSize',24)
ax = gca;
ax.XAxis.FontSize = 24;
xtickangle(ax, 75);
saveas(gcf,['../FiguresUsedinthePaper/',filename,'_Wass2'],'png')
saveas(gcf,['../FiguresUsedinthePaper/',filename,'_Wass2'],'fig')
saveas(gcf,['../FiguresUsedinthePaper/',filename,'_Wass2'],'eps')

% 
% figure
% bar(STi)
% title(strcat('Ground Truth; rho=',num2str(rho)))
% xlabel('Input')
% ylabel('\tau_i')
% set(gca,'FontSize',24)
% set(gcf,'WindowState','maximized')
% saveas(gcf,['../FiguresUsedinthePaper/',filename,'_GroundTruth'],'fig')
% saveas(gcf,['../FiguresUsedinthePaper/',filename,'_GroundTruth'],'eps')
% saveas(gcf,['../FiguresUsedinthePaper/',filename,'_GroundTruth'],'jpg')
% saveas(gcf,['../FiguresUsedinthePaper/',filename,'_GroundTruth'],'png')
% writematrix(STi,filename)
%% Data visualization
figure
histogram(y, 'Normalization', 'pdf');
hold on;
[c0,y1]=ksdensity(y);
plot(y1,c0,"Color",'b','LineWidth',2)
% Titles and labels
title('Count of total rental bikes', 'FontSize', 24);
xlabel('Rental Count');
ylabel('Probability Density');
legend({'Histogram', 'Estimated Density'}, 'Location', 'best',FontSize=24);
grid on;

%% visualize X
% Number of features
numVars = size(x,2);
% Create a figure
figure;
numRows = ceil(sqrt(numVars));
numCols = ceil(numVars / numRows);
for i = 1:numVars
    subplot(numRows, numCols, i);
    colData = x(:, i);   
    if isnumeric(colData)
        histogram(colData, 'Normalization', 'pdf');
        title(labels{i}, 'Interpreter', 'none', 'FontSize', 10);
    else
        % Skip non-numeric variables (optional: show message)
        title(labels{i}, 'Interpreter', 'none', 'FontSize', 8);
    end
end
sgtitle('Histograms of Features');
%% Ensemble Bagged Trees
[BagTre,validationAccuracyBT]=trainBikeSharingEnsembleBT(x,y)

validationAccuracyBT

mdl3=@(x) BagTre.predictFcn(x)

% RF: Variable Importance over Replicates
namemdl='Ensamble Bagged Trees';

YtestKS=1;
indices=[4];
for jj=1:Rep
    if jj==Rep
        Yplot=1;
    else
        Yplot=0;
    end
[nuBTUnraux,tauprime,TestResultsUnrAux]=nuBrei(x,y,mdl3,namemdl,Yplot,indices,YtestKS,'density');
nuBTUnr(jj,:)=nuBTUnraux;
BT_TestResultsUnr1(jj,:)=TestResultsUnrAux(1,:);
BT_TestResultsUnr2(jj,:)=TestResultsUnrAux(2,:);
BT_TestResultsUnr3(jj,:)=TestResultsUnrAux(3,:);
[nuBTGCMRaux,BT_tauprimeGCMR,TestResultsGCMRAux]=nuGCMRdiscr(x,y,mdl3,namemdl,Yplot,indices,YtestKS,'density');
nuBTGCMR(jj,:)=nuBTGCMRaux;
BT_TestResultsGCMR1(jj,:)=TestResultsGCMRAux(1,:);
BT_TestResultsGCMR2(jj,:)=TestResultsGCMRAux(2,:);
BT_TestResultsGCMR3(jj,:)=TestResultsGCMRAux(3,:);
[nuBTKnockaux,tauprimerestrKnock,TestResultsGKnockAux]=fnuGKnock(x,y,mdl3,namemdl,Yplot,indices,YtestKS,'density');
nuBTKnock(jj,:)=nuBTKnockaux;
BT_TestResultsGKnock1(jj,:)=TestResultsGKnockAux(1,:);
BT_TestResultsGKnock2(jj,:)=TestResultsGKnockAux(2,:);
BT_TestResultsGKnock3(jj,:)=TestResultsGKnockAux(3,:);
waitbar(jj/Rep);
end
%%
AABT=[mean(nuBTUnr)/(2*VY); mean(nuBTGCMR)/(2*VY); mean(nuBTKnock)/(2*VY)];
figure
b = bar(AABT'); b(1).FaceColor = [0 0 1]; b(2).FaceColor = [1 0 0]; b(3).FaceColor=[0 1 0];
title('Ensemble Bagged Trees, $\nu_j$, $\nu_j^{GCMR}$ and $\nu_j^{GKnock}$','FontSize',30,'Interpreter','latex')
legend('$\nu_j$', '$\nu_j^{GCMR}$', '$\nu_j^{GKnock}$','FontSize',28,'Interpreter','latex')
xticklabels(labels)
set(gca,'TickLabelInterpreter','none')
xtickangle(90); % Set orientation to vertical
set(gca,'FontSize',18)
set(gcf,'WindowState','maximized')
%%
saveas(gcf,['../FiguresUsedinthePaper/',filename,'_BT_Breiman'],'fig')
saveas(gcf,['../FiguresUsedinthePaper/',filename,'_BT_Breiman'],'eps')
saveas(gcf,['../FiguresUsedinthePaper/',filename,'_BT_Breiman'],'jpg')
saveas(gcf,['../FiguresUsedinthePaper/',filename,'_BT_Breiman'],'png')

%% Neural Network

[NNc,validationAccuracyNN]=trainBikeSharingNN(x,y)

KfoldNNValAcc=validationAccuracyNN;

mdl5=@(x) NNc.predictFcn(x)

% RF: Variable Importance over Replicates
namemdl='NN';

YtestKS=1;
indices=[46];
for jj=1:Rep
    if jj==Rep
        Yplot=1;
    else
        Yplot=0;
    end
[nuNNUnraux,tauprime,TestResultsUnrAux]=nuBrei(x,y,mdl5,namemdl,Yplot,indices,YtestKS,'density');
nuNNUnr(jj,:)=nuNNUnraux;
NN_TestResultsUnr1(jj,:)=TestResultsUnrAux(1,:);
NN_TestResultsUnr2(jj,:)=TestResultsUnrAux(2,:);
NN_TestResultsUnr3(jj,:)=TestResultsUnrAux(3,:);
[nuNNGCMRaux,NN_tauprimeGCMR,TestResultsGCMRAux]=nuGCMR(x,y,mdl5,namemdl,Yplot,indices,YtestKS,'density');
nuNNGCMR(jj,:)=nuNNGCMRaux;
NN_TestResultsGCMR1(jj,:)=TestResultsGCMRAux(1,:);
NN_TestResultsGCMR2(jj,:)=TestResultsGCMRAux(2,:);
NN_TestResultsGCMR3(jj,:)=TestResultsGCMRAux(3,:);
[nuNNKnockaux,tauprimerestrKnock,TestResultsGKnockAux]=fnuGKnock(x,y,mdl5,namemdl,Yplot,indices,YtestKS,'density');
nuNNKnock(jj,:)=nuNNKnockaux;
NN_TestResultsGKnock1(jj,:)=TestResultsGKnockAux(1,:);
NN_TestResultsGKnock2(jj,:)=TestResultsGKnockAux(2,:);
NN_TestResultsGKnock3(jj,:)=TestResultsGKnockAux(3,:);
waitbar(jj/Rep);
end

AANN=[mean(nuNNUnr)/(2*VY); mean(nuNNGCMR)/(2*VY); mean(nuNNKnock)/(2*VY)];
figure
b = bar(AANN'); b(1).FaceColor = [0 0 1];b(2).FaceColor = [1 0 0];b(3).FaceColor=[0 1 0];
title('NN, $\nu_j$, $\nu_j^{GCMR}$ and $\nu_j^{GKnock}$','FontSize',30,'Interpreter','latex')
legend('$\nu_j$', '$\nu_j^{GCMR}$', '$\nu_j^{GKnock}$','FontSize',28,'Interpreter','latex')
xticklabels(labels)
set(gca,'TickLabelInterpreter','none')
xtickangle(90); % Set orientation to vertical
set(gca,'FontSize',18)
set(gcf,'WindowState','maximized')
saveas(gcf,['../FiguresUsedinthePaper/',filename,'_NN_Breiman'],'fig')
saveas(gcf,['../FiguresUsedinthePaper/',filename,'_NN_Breiman'],'eps')
saveas(gcf,['../FiguresUsedinthePaper/',filename,'_NN_Breiman'],'jpg')
saveas(gcf,['../FiguresUsedinthePaper/',filename,'_NN_Breiman'],'png')

save(filename,'AANN','AABT','Wass2','KfoldNNValAcc','validationAccuracyBT')