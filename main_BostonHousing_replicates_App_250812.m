close all;clearvars;clc

%Alreadytrained=input('Already Trained?; 1 if true, 0 if false ');
Rep=input('Number of Replicates ');


data=readmatrix('Housing.xlsx');
x=data(:,1:end-2);
[n,d]=size(x);
y=data(:,end-1);

labels={'CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT'};


%writetable(Data, 'bikesharing_data.csv');
%%
filename = matlab.desktop.editor.getActiveFilename;
[~, filename, ~] = fileparts(filename)
%netname=strcat('trainednet',filename);
%netnamemat=strcat(netname,'.mat')
%% Calculate wassersi -  not run now
VY=var(y);
[W]=wassersi(x,y,12);
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

%% Loadn Models
load('BostonHousingModels.mat')

%% XGBoost

mdl3=@(x) BostonXGBoost.predictFcn(x)

% RF: Variable Importance over Replicates
namemdl='Ensamble Boosted Trees';

YtestKS=1;
indices=[6,13];
for jj=1:Rep
    namemdl
    jj
    if jj==Rep
        Yplot=1;
    else
        Yplot=0;
    end
[nuBTUnraux,tauprime,TestResultsUnrAux]=nuBrei(x,y,mdl3,namemdl,Yplot,indices,YtestKS,'density');
nuBTUnrXGBoost(jj,:)=nuBTUnraux;
BT_TestResultsUnr1(jj,:)=TestResultsUnrAux(1,:);
BT_TestResultsUnr2(jj,:)=TestResultsUnrAux(2,:);
BT_TestResultsUnr3(jj,:)=TestResultsUnrAux(3,:);
[nuBTGCMRaux,BT_tauprimeGCMR,TestResultsGCMRAux]=nuGCMRdiscr(x,y,mdl3,namemdl,Yplot,indices,YtestKS,'density');
nuBTGCMRXGBoost(jj,:)=nuBTGCMRaux;
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
AABT=[mean(nuBTUnrXGBoost)/(2*VY); mean(nuBTGCMRXGBoost)/(2*VY); mean(nuBTKnock)/(2*VY)];
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

%% GPR

%[NNc,validationAccuracyNN]=trainBikeSharingNN(x,y)

%KfoldNNValAcc=validationAccuracyNN;

mdl5=@(x) BostonGPR.predictFcn(x)

% RF: Variable Importance over Replicates
namemdl='GPR';

YtestKS=1;
indices=[6,13];
for jj=1:Rep
    namemdl
    jj
    if jj==Rep
        Yplot=1;
    else
        Yplot=0;
    end
[nuGPRUnraux,tauprime,TestResultsUnrAux]=nuBrei(x,y,mdl5,namemdl,Yplot,indices,YtestKS,'density');
nuGPRUnr(jj,:)=nuGPRUnraux;
GPR_TestResultsUnr1(jj,:)=TestResultsUnrAux(1,:);
GPR_TestResultsUnr2(jj,:)=TestResultsUnrAux(2,:);
GPR_TestResultsUnr3(jj,:)=TestResultsUnrAux(3,:);
[nuGPRGCMRaux,GPR_tauprimeGCMR,TestResultsGCMRAux]=nuGCMR(x,y,mdl5,namemdl,Yplot,indices,YtestKS,'density');
nuGPRGCMR(jj,:)=nuGPRGCMRaux;
GPR_TestResultsGCMR1(jj,:)=TestResultsGCMRAux(1,:);
GPR_TestResultsGCMR2(jj,:)=TestResultsGCMRAux(2,:);
GPR_TestResultsGCMR3(jj,:)=TestResultsGCMRAux(3,:);
[nuGPRKnockaux,tauprimerestrKnock,TestResultsGKnockAux]=fnuGKnock(x,y,mdl5,namemdl,Yplot,indices,YtestKS,'density');
nuGPRKnock(jj,:)=nuGPRKnockaux;
GPR_TestResultsGKnock1(jj,:)=TestResultsGKnockAux(1,:);
GPR_TestResultsGKnock2(jj,:)=TestResultsGKnockAux(2,:);
GPR_TestResultsGKnock3(jj,:)=TestResultsGKnockAux(3,:);
waitbar(jj/Rep);
end

AAGPR=[mean(nuGPRUnr)/(2*VY); mean(nuGPRGCMR)/(2*VY); mean(nuGPRKnock)/(2*VY)];
figure
b = bar(AAGPR'); b(1).FaceColor = [0 0 1];b(2).FaceColor = [1 0 0];b(3).FaceColor=[0 1 0];
title('GPR, $\nu_j$, $\nu_j^{GCMR}$ and $\nu_j^{GKnock}$','FontSize',30,'Interpreter','latex')
legend('$\nu_j$', '$\nu_j^{GCMR}$', '$\nu_j^{GKnock}$','FontSize',28,'Interpreter','latex')
xticklabels(labels)
set(gca,'TickLabelInterpreter','none')
xtickangle(90); % Set orientation to vertical
set(gca,'FontSize',18)
set(gcf,'WindowState','maximized')
saveas(gcf,['../FiguresUsedinthePaper/',filename,'_GPR_Breiman'],'fig')
saveas(gcf,['../FiguresUsedinthePaper/',filename,'_GPR_Breiman'],'eps')
saveas(gcf,['../FiguresUsedinthePaper/',filename,'_GPR_Breiman'],'jpg')
saveas(gcf,['../FiguresUsedinthePaper/',filename,'_GPR_Breiman'],'png')


%% GPR

%[NNc,validationAccuracyNN]=trainBikeSharingNN(x,y)

%KfoldNNValAcc=validationAccuracyNN;

mdl5=@(x) BostonSVM.predictFcn(x)

% RF: Variable Importance over Replicates
namemdl='SVM';

YtestKS=1;
indices=[6,13];
for jj=1:Rep
    namemdl
    jj
    if jj==Rep
        Yplot=1;
    else
        Yplot=0;
    end
[nuSVMUnraux,tauprime,TestResultsUnrAux]=nuBrei(x,y,mdl5,namemdl,Yplot,indices,YtestKS,'density');
nuSVMUnr(jj,:)=nuSVMUnraux;
SVM_TestResultsUnr1(jj,:)=TestResultsUnrAux(1,:);
SVM_TestResultsUnr2(jj,:)=TestResultsUnrAux(2,:);
SVM_TestResultsUnr3(jj,:)=TestResultsUnrAux(3,:);
[nuSVMGCMRaux,SVM_tauprimeGCMR,TestResultsGCMRAux]=nuGCMR(x,y,mdl5,namemdl,Yplot,indices,YtestKS,'density');
nuSVMGCMR(jj,:)=nuSVMGCMRaux;
SVM_TestResultsGCMR1(jj,:)=TestResultsGCMRAux(1,:);
SVM_TestResultsGCMR2(jj,:)=TestResultsGCMRAux(2,:);
SVM_TestResultsGCMR3(jj,:)=TestResultsGCMRAux(3,:);
[nuSVMKnockaux,tauprimerestrKnock,TestResultsGKnockAux]=fnuGKnock(x,y,mdl5,namemdl,Yplot,indices,YtestKS,'density');
nuSVMKnock(jj,:)=nuSVMKnockaux;
SVM_TestResultsGKnock1(jj,:)=TestResultsGKnockAux(1,:);
SVM_TestResultsGKnock2(jj,:)=TestResultsGKnockAux(2,:);
SVM_TestResultsGKnock3(jj,:)=TestResultsGKnockAux(3,:);
waitbar(jj/Rep);
end

AASVM=[mean(nuSVMUnr)/(2*VY); mean(nuSVMGCMR)/(2*VY); mean(nuSVMKnock)/(2*VY)];
figure
b = bar(AASVM'); b(1).FaceColor = [0 0 1];b(2).FaceColor = [1 0 0];b(3).FaceColor=[0 1 0];
title('SVM, $\nu_j$, $\nu_j^{GCMR}$ and $\nu_j^{GKnock}$','FontSize',30,'Interpreter','latex')
legend('$\nu_j$', '$\nu_j^{GCMR}$', '$\nu_j^{GKnock}$','FontSize',28,'Interpreter','latex')
xticklabels(labels)
set(gca,'TickLabelInterpreter','none')
xtickangle(90); % Set orientation to vertical
set(gca,'FontSize',18)
set(gcf,'WindowState','maximized')
saveas(gcf,['../FiguresUsedinthePaper/',filename,'_SVM_Breiman'],'fig')
saveas(gcf,['../FiguresUsedinthePaper/',filename,'_SVM_Breiman'],'eps')
saveas(gcf,['../FiguresUsedinthePaper/',filename,'_SVM_Breiman'],'jpg')
saveas(gcf,['../FiguresUsedinthePaper/',filename,'_SVM_Breiman'],'png')


save(filename)