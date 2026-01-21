close all;clearvars;clc
%
%Alreadytrained=input('Already Trained?; 1 if true, 0 if false ');
Rep=input('Number of Replicates: ');
%%
load("spambase_data.mat")
load('randperm_spambase.mat')
%removecolumns = {};
%Data(:,removecolumns) = [];
summary(Data)
Data=Data(aa,:);
% Output variable
targetVar = 'spam';

% Separate predictors and target
y = Data.(targetVar);  % Extract target variable (as vector)
x = removevars(Data, targetVar);  % Remove 'cnt' column from table
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
important_indices = [52, 53, 21, 56, 55, 16, 57, 7, 5, 19, 24, 25, 3, 23, 17];
figure
bar(Wass2)
%title('Wasserstein Importance ($\iota^{W_2^2}$)','Interpreter','Latex')
[aaa,bbb]=sort(important_indices,'ascend');
%xticks(aaa'-1)
%xticklabels(labels)
xlabel('Feature X_i, i=1,2,...,57')
set(gca,'TickLabelInterpreter','none')
set(gca,'FontSize',28)
ylabel('$\hat{\iota}^{W_2^2}$','Interpreter','Latex','FontSize',34)
saveas(gcf,['../FiguresUsedinthePaper/',filename,'Wass2'],'png')
saveas(gcf,['../FiguresUsedinthePaper/',filename,'Wass2'],'fig')

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


%% Load Models
load('SpambaseModels.mat')

%% XGBoost

mdl3=@(x) SpambaseRF.predictFcn(x)

% RF: Variable Importance over Replicates
namemdl='Random Forest';

YtestKS=1;
indices=[25];
tic
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
TRF=toc
%
AABT=[mean(nuBTUnrXGBoost)/(2*VY); mean(nuBTGCMRXGBoost)/(2*VY); mean(nuBTKnock)/(2*VY)];
figure
b = bar(AABT'); b(1).FaceColor = [0 0 1]; b(2).FaceColor = [1 0 0]; b(3).FaceColor=[0 1 0];
title('Random Forest')
legend('$\nu_j$', '$\nu_j^{GCMR}$', '$\nu_j^{GKnock}$','FontSize',28,'Interpreter','latex')
xticklabels(labels)
set(gca,'TickLabelInterpreter','none')
xtickangle(90); % Set orientation to vertical
set(gca,'FontSize',24)
set(gcf,'WindowState','maximized')
%%
saveas(gcf,['../FiguresUsedinthePaper/',filename,'_BT_Breiman'],'fig')
saveas(gcf,['../FiguresUsedinthePaper/',filename,'_BT_Breiman'],'eps')
saveas(gcf,['../FiguresUsedinthePaper/',filename,'_BT_Breiman'],'jpg')
saveas(gcf,['../FiguresUsedinthePaper/',filename,'_BT_Breiman'],'png')

%% NN

%[NNc,validationAccuracyNN]=trainBikeSharingNN(x,y)

%KfoldNNValAcc=validationAccuracyNN;

mdl5=@(x) SpambaseNN.predictFcn(x)

% RF: Variable Importance over Replicates
namemdl='Artificial NN';

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


%% SVM

%[NNc,validationAccuracyNN]=trainBikeSharingNN(x,y)

%KfoldNNValAcc=validationAccuracyNN;

mdl5=@(x) SpambaseSVM.predictFcn(x)

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

%%
AASVM=[mean(nuSVMUnr)/(2*VY); mean(nuSVMGCMR)/(2*VY); mean(nuSVMKnock)/(2*VY)];
figure
b = bar(AASVM'); b(1).FaceColor = [0 0 1];b(2).FaceColor = [1 0 0];b(3).FaceColor=[0 1 0];
title('SVM')
legend('$\nu_j$', '$\nu_j^{GCMR}$', '$\nu_j^{GKnock}$','FontSize',28,'Interpreter','latex')
set(gca,'TickLabelInterpreter','none')
xtickangle(90); % Set orientation to vertical
set(gca,'FontSize',24)
set(gcf,'WindowState','maximized')
saveas(gcf,['../FiguresUsedinthePaper/',filename,'_SVM_Breiman'],'fig')
saveas(gcf,['../FiguresUsedinthePaper/',filename,'_SVM_Breiman'],'eps')
saveas(gcf,['../FiguresUsedinthePaper/',filename,'_SVM_Breiman'],'jpg')
saveas(gcf,['../FiguresUsedinthePaper/',filename,'_SVM_Breiman'],'png')


%% One Figure
figure
subplot(3,1,1)
b = bar(AABT'); b(1).FaceColor = [0 0 1]; b(2).FaceColor = [1 0 0]; b(3).FaceColor=[0 1 0];
title('Random Forest')
%legend('$\nu_j$', '$\nu_j^{GCMR}$', '$\nu_j^{GKnock}$','FontSize',28,'Interpreter','latex')
xticks([1:57])
set(gca,'TickLabelInterpreter','none')
xtickangle(90); % Set orientation to vertical
set(gca,'FontSize',24)
set(gcf,'WindowState','maximized')

subplot(3,1,2)
b = bar(AANN'); b(1).FaceColor = [0 0 1];b(2).FaceColor = [1 0 0];b(3).FaceColor=[0 1 0];
title('Artificial NN')
legend('$\nu_j$', '$\nu_j^{GCMR}$', '$\nu_j^{GKnock}$','FontSize',28,'Interpreter','latex')
xticks([1:57])
%xticklabels(labels)
set(gca,'TickLabelInterpreter','none')
xtickangle(90); % Set orientation to vertical
set(gca,'FontSize',24)
set(gcf,'WindowState','maximized')


subplot(3,1,3)
AASVM=[mean(nuSVMUnr)/(2*VY); mean(nuSVMGCMR)/(2*VY); mean(nuSVMKnock)/(2*VY)];
b = bar(AASVM'); b(1).FaceColor = [0 0 1];b(2).FaceColor = [1 0 0];b(3).FaceColor=[0 1 0];
title('SVM')
%legend('$\nu_j$', '$\nu_j^{GCMR}$', '$\nu_j^{GKnock}$','FontSize',28,'Interpreter','latex')
xticks([1:57])
set(gca,'TickLabelInterpreter','none')
xtickangle(90); % Set orientation to vertical
set(gca,'FontSize',24)
set(gcf,'WindowState','maximized')

saveas(gcf,['../FiguresUsedinthePaper/',filename,'_All_Breiman'],'fig')
saveas(gcf,['../FiguresUsedinthePaper/',filename,'_All_Breiman'],'eps')
saveas(gcf,['../FiguresUsedinthePaper/',filename,'_All_Breiman'],'jpg')
saveas(gcf,['../FiguresUsedinthePaper/',filename,'_All_Breiman'],'png')

%%
save(filename)