close all;clearvars;clc
Rep=100;

%%
filename = matlab.desktop.editor.getActiveFilename;
[~, filename, ~] = fileparts(filename);
%netname=strcat('trainednet',filename);
%netnamemat=strcat(netname,'.mat')
%%
% synthetic data generation

rng("default")
rho=0.5;
d=500;
labels = arrayfun(@(k) sprintf('X_{%d}', k), 1:d, 'UniformOutput', false);
Sigma=zeros(d,d);
for i=1:d
    for j=1:d
        if i==j
        Sigma(i,j)=1;
        else
        Sigma(i,j)=rho;
        end
    end
end

mu=0*ones(1,d);
N=50000;
x=mvnrnd(mu,Sigma,N);
x=normcdf(x);

mdl=@(x) Model500(x);
y=mdl(x);
VY=var(y)
[W,d,nn]=wassersi(x,y,50);

%[Si1,STi1]=ihssi(d,N,mdl,@(u)u)
[Si,STi]=windsi(Sigma,N,mdl,@(u)u);

%[Si,STi]=windsi(Sigma,N,mdl,@normcdf)

Wass1=W.W1;
Wass2=W.W22/(2*var(y));
Wass8=W.W8;

figure
bar(Wass2)
title('Wasserstein 2')
xlabel('Input')
ylabel('\tau_i')
xticklabels(labels)

figure
bar(STi)
title(strcat('Ground Truth; rho=',num2str(rho)))
xlabel('Input')
ylabel('\tau_i')
xticklabels(labels)
set(gca,'FontSize',24)
set(gcf,'WindowState','maximized')
saveas(gcf,['../FiguresUsedinthePaper/',filename,'_GroundTruth_50000'],'fig')
saveas(gcf,['../FiguresUsedinthePaper/',filename,'_GroundTruth_50000'],'eps')
%saveas(gcf,'../FiguresUsedinthePaper\Model500_GroundTruth_50000_0','.jpg')
saveas(gcf,['../FiguresUsedinthePaper/',filename,'_GroundTruth_50000'],'png')

%writematrix(STi,filename,"FileType", 'spreadsheet','Sheet','GroundTruth')
%%

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
%% Feature Importance: Load Models
load('Model500_Models.mat')

%% Tree see Model500ModelPerformance.docx
mdl3=@(x) Model500_NN.predictFcn(x)

% RF: Variable Importance over Replicates
namemdl='NN';

YtestKS=1;
indices=[2];
for jj=1:Rep
    if jj==Rep
        Yplot=1;
    else
        Yplot=0;
    end
[nuNNUnraux,tauprime,TestResultsUnrAux]=nuBrei(x,y,mdl3,namemdl,Yplot,indices,YtestKS,'density');
nuNNUnr(jj,:)=nuNNUnraux;
NN_TestResultsUnr1(jj,:)=TestResultsUnrAux(1,:);
NN_TestResultsUnr2(jj,:)=TestResultsUnrAux(2,:);
NN_TestResultsUnr3(jj,:)=TestResultsUnrAux(3,:);
[nuNNGCMRaux,NN_tauprimeGCMR,TestResultsGCMRAux]=nuGCMR(x,y,mdl3,namemdl,Yplot,indices,YtestKS,'density');
nuNNGCMR(jj,:)=nuNNGCMRaux;
NN_TestResultsGCMR1(jj,:)=TestResultsGCMRAux(1,:);
NN_TestResultsGCMR2(jj,:)=TestResultsGCMRAux(2,:);
NN_TestResultsGCMR3(jj,:)=TestResultsGCMRAux(3,:);
[nuNNKnockaux,tauprimerestrKnock,TestResultsGKnockAux]=fnuGKnock(x,y,mdl3,namemdl,Yplot,indices,YtestKS,'density');
nuNNKnock(jj,:)=nuNNKnockaux;
NN_TestResultsGKnock1(jj,:)=TestResultsGKnockAux(1,:);
NN_TestResultsGKnock2(jj,:)=TestResultsGKnockAux(2,:);
NN_TestResultsGKnock3(jj,:)=TestResultsGKnockAux(3,:);
waitbar(jj/Rep);
end
%%
AANN=[STi;mean(nuNNUnr)/(2*VY); mean(nuNNGCMR)/(2*VY); mean(nuNNKnock)/(2*VY)];
figure
b = bar(AANN'); b(1).FaceColor = [0 0 0]; b(2).FaceColor = [0 0 1]; b(3).FaceColor = [1 0 0]; b(4).FaceColor=[0 1 0];
title('Artificial NN, $\nu_j$, $\nu_j^{GCMR}$ and $\nu_j^{GKnock}$','FontSize',30,'Interpreter','latex')
legend('$\nu_j$', '$\nu_j^{GCMR}$', '$\nu_j^{GKnock}$','FontSize',28,'Interpreter','latex')
xticklabels(labels)
%set(gca,'TickLabelInterpreter','none')
xtickangle(90); % Set orientation to vertical
set(gca,'FontSize',18)
set(gcf,'WindowState','maximized')
%%
saveas(gcf,['../FiguresUsedinthePaper/',filename,'_NN_Breiman'],'fig')
saveas(gcf,['../FiguresUsedinthePaper/',filename,'_NN_Breiman'],'eps')
saveas(gcf,['../FiguresUsedinthePaper/',filename,'_NN_Breiman'],'jpg')
saveas(gcf,['../FiguresUsedinthePaper/',filename,'_NN_Breiman'],'png')



%% Tree see Model_500_Performance.docx
mdl4=@(x) Model500_EBT.predictFcn(x)

% RF: Variable Importance over Replicates
namemdl='Ensemble BT';

YtestKS=1;
indices=[2];
for jj=1:Rep
    if jj==Rep
        Yplot=1;
    else
        Yplot=0;
    end
[nuBTUnraux,tauprime,TestResultsUnrAux]=nuBrei(x,y,mdl4,namemdl,Yplot,indices,YtestKS,'density');
nuBTUnr(jj,:)=nuBTUnraux;
BT_TestResultsUnr1(jj,:)=TestResultsUnrAux(1,:);
BT_TestResultsUnr2(jj,:)=TestResultsUnrAux(2,:);
BT_TestResultsUnr3(jj,:)=TestResultsUnrAux(3,:);
[nuBTGCMRaux,BT_tauprimeGCMR,TestResultsGCMRAux]=nuGCMRdiscr(x,y,mdl4,namemdl,Yplot,indices,YtestKS,'density');
nuBTGCMR(jj,:)=nuBTGCMRaux;
BT_TestResultsGCMR1(jj,:)=TestResultsGCMRAux(1,:);
BT_TestResultsGCMR2(jj,:)=TestResultsGCMRAux(2,:);
BT_TestResultsGCMR3(jj,:)=TestResultsGCMRAux(3,:);
[nuBTKnockaux,tauprimerestrKnock,TestResultsGKnockAux]=fnuGKnock(x,y,mdl4,namemdl,Yplot,indices,YtestKS,'density');
nuBTKnock(jj,:)=nuBTKnockaux;
BT_TestResultsGKnock1(jj,:)=TestResultsGKnockAux(1,:);
BT_TestResultsGKnock2(jj,:)=TestResultsGKnockAux(2,:);
BT_TestResultsGKnock3(jj,:)=TestResultsGKnockAux(3,:);
waitbar(jj/Rep);
end
%%
AABT=[STi;mean(nuBTUnr)/(2*VY); mean(nuBTGCMR)/(2*VY); mean(nuBTKnock)/(2*VY)];
figure
b = bar(AABT');  b(1).FaceColor = [0 0 0]; b(2).FaceColor = [0 0 1]; b(3).FaceColor = [1 0 0]; b(4).FaceColor=[0 1 0];
title('BT, $\nu_j$, $\nu_j^{GCMR}$ and $\nu_j^{GKnock}$','FontSize',30,'Interpreter','latex')
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



save(filename,'AANN','AALR','AABT','STi')