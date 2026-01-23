close all;clearvars;clc
Rep=100;

%%
Alreadytrained=1;%input('Already Trained?; 1 if true, 0 if false ');
Rep=100%input('Number of Replicates ');
d=20;
labels = arrayfun(@(k) sprintf('X_{%d}', k), 1:d, 'UniformOutput', false);

filename = matlab.desktop.editor.getActiveFilename;
[~, filename, ~] = fileparts(filename);
%netname='trainednetMainMorris99_Copula_50000_replicates';
%netnamemat=strcat(netname,'.mat')
% synthetic data generation
% important decreases with a(i)

%%
% synthetic data generation
rng("default")
rho=0.5;
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

mdl=@(x) moon(x);
y=mdl(x);
VY=var(y)
[W]=wassersi(x,y,15);

%[Si1,STi1]=ihssi(d,N,mdl,@(u)u)
[Si,STi]=windsi(Sigma,N,mdl,@(u)u);
writematrix(STi,'MoonResults.xlsx','FileType','spreadsheet','Sheet','STi')

%[Si,STi]=windsi(Sigma,N,mdl,@normcdf)

Wass1=W.W1;
Wass2=W.W22/(2*var(y));
Wass8=W.W8;

figure
bar(Wass2)
title('Wasserstein 2')
xlabel('Input')
xticks([1:d])
xticklabels(labels)
ylabel('\tau_i')
set(gca,'FontSize',24)
set(gcf,'WindowState','maximized')

figure
bar(STi)
title(strcat('Ground Truth; rho=',num2str(rho)))
xlabel('Input')
ylabel('\tau_i')
xticks([1:d])
xticklabels(labels)
set(gca,'FontSize',24)
set(gcf,'WindowState','maximized')
saveas(gcf,['../FiguresUsedinthePaper/',filename,'_GroundTruth_50000'],'fig')
saveas(gcf,['../FiguresUsedinthePaper/',filename,'_GroundTruth_50000'],'eps')
%saveas(gcf,'../FiguresUsedinthePaper\Model99_GroundTruth_50000_0','.jpg')
saveas(gcf,['../FiguresUsedinthePaper/',filename,'_GroundTruth_50000'],'png')

writematrix(STi,filename,"FileType", 'spreadsheet','Sheet','GroundTruth')
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
%% Load Models
load('Moon_Models.mat')


%% Tree see IshigamiModelPerformance.docx
%[NN,validationAccuracyNN]=trainMoonNN(x,y)


mdl3=@(x) MoonNN.predictFcn(x)

% RF: Variable Importance over Replicates
namemdl='Neural Network';

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
[nuNNGCMRaux,NN_tauprimeGCMR,TestResultsGCMRAux]=nuGCMRdiscr(x,y,mdl3,namemdl,Yplot,indices,YtestKS,'density');
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
set(gca,'TickLabelInterpreter','none')
xtickangle(90); % Set orientation to vertical
set(gca,'FontSize',18)
set(gcf,'WindowState','maximized')

saveas(gcf,['../FiguresUsedinthePaper/',filename,'_NN_Breiman'],'fig')
saveas(gcf,['../FiguresUsedinthePaper/',filename,'_NN_Breiman'],'eps')
saveas(gcf,['../FiguresUsedinthePaper/',filename,'_NN_Breiman'],'jpg')
saveas(gcf,['../FiguresUsedinthePaper/',filename,'_NN_Breiman'],'png')

writematrix(AANN,'MoonResults.xlsx','FileType','spreadsheet','Sheet','NeuralNetwork')

%% Tree see IshigamiModelPerformance.docx
%[SVM,validationAccuracySVM]=trainMoonSVM(x,y)

%validationAccuracySVM

mdl5=@(x) MoonSVM.predictFcn(x)

% RF: Variable Importance over Replicates
namemdl='SVM';

YtestKS=1;
indices=[2];
for jj=1:Rep
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
[nuSVMGCMRaux,SVM_tauprimeGCMR,TestResultsGCMRAux]=nuGCMRdiscr(x,y,mdl5,namemdl,Yplot,indices,YtestKS,'density');
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
AASVM=[STi;mean(nuSVMUnr)/(2*VY); mean(nuSVMGCMR)/(2*VY); mean(nuSVMKnock)/(2*VY)];
figure
b = bar(AASVM');  b(1).FaceColor = [0 0 0]; b(2).FaceColor = [0 0 1]; b(3).FaceColor = [1 0 0]; b(4).FaceColor=[0 1 0];
title('SVMs, $\nu_j$, $\nu_j^{GCMR}$ and $\nu_j^{GKnock}$','FontSize',30,'Interpreter','latex')
legend('$\nu_j$', '$\nu_j^{GCMR}$', '$\nu_j^{GKnock}$','FontSize',28,'Interpreter','latex')
xticklabels(labels)
set(gca,'TickLabelInterpreter','none')
xtickangle(90); % Set orientation to vertical
set(gca,'FontSize',18)
set(gcf,'WindowState','maximized')
%%
saveas(gcf,['../FiguresUsedinthePaper/',filename,'_SVM_Breiman'],'fig')
saveas(gcf,['../FiguresUsedinthePaper/',filename,'_SVM_Breiman'],'eps')
saveas(gcf,['../FiguresUsedinthePaper/',filename,'_SVM_Breiman'],'jpg')
saveas(gcf,['../FiguresUsedinthePaper/',filename,'_SVM_Breiman'],'png')

writematrix(AASVM,'MoonResults.xlsx','FileType','spreadsheet','Sheet','SVM')

%% Tree see IshigamiModelPerformance.docx
%[GPR,validationAccuracyGPR]=trainMoonGPR(x,y)

%validationAccuracyGPR

mdl4=@(x) MoonGPR.predictFcn(x)

% RF: Variable Importance over Replicates
namemdl='Gaussia Process Regression';

YtestKS=1;
indices=[2];
for jj=1:Rep
    if jj==Rep
        Yplot=1;
    else
        Yplot=0;
    end
[nuGPRUnraux,tauprime,TestResultsUnrAux]=nuBrei(x,y,mdl4,namemdl,Yplot,indices,YtestKS,'density');
nuGPRUnr(jj,:)=nuGPRUnraux;
GPR_TestResultsUnr1(jj,:)=TestResultsUnrAux(1,:);
GPR_TestResultsUnr2(jj,:)=TestResultsUnrAux(2,:);
GPR_TestResultsUnr3(jj,:)=TestResultsUnrAux(3,:);
[nuGPRGCMRaux,GPR_tauprimeGCMR,TestResultsGCMRAux]=nuGCMRdiscr(x,y,mdl4,namemdl,Yplot,indices,YtestKS,'density');
nuGPRGCMR(jj,:)=nuGPRGCMRaux;
GPR_TestResultsGCMR1(jj,:)=TestResultsGCMRAux(1,:);
GPR_TestResultsGCMR2(jj,:)=TestResultsGCMRAux(2,:);
GPR_TestResultsGCMR3(jj,:)=TestResultsGCMRAux(3,:);
[nuGPRKnockaux,tauprimerestrKnock,TestResultsGKnockAux]=fnuGKnock(x,y,mdl4,namemdl,Yplot,indices,YtestKS,'density');
nuGPRKnock(jj,:)=nuGPRKnockaux;
GPR_TestResultsGKnock1(jj,:)=TestResultsGKnockAux(1,:);
GPR_TestResultsGKnock2(jj,:)=TestResultsGKnockAux(2,:);
GPR_TestResultsGKnock3(jj,:)=TestResultsGKnockAux(3,:);
waitbar(jj/Rep);
end
%%
AAGPR=[STi;mean(nuGPRUnr)/(2*VY); mean(nuGPRGCMR)/(2*VY); mean(nuGPRKnock)/(2*VY)];
figure
b = bar(AAGPR');  b(1).FaceColor = [0 0 0]; b(2).FaceColor = [0 0 1]; b(3).FaceColor = [1 0 0]; b(4).FaceColor=[0 1 0];
title('GPR, $\nu_j$, $\nu_j^{GCMR}$ and $\nu_j^{GKnock}$','FontSize',30,'Interpreter','latex')
legend('$\nu_j$', '$\nu_j^{GCMR}$', '$\nu_j^{GKnock}$','FontSize',28,'Interpreter','latex')
xticklabels(labels)
set(gca,'TickLabelInterpreter','none')
xtickangle(90); % Set orientation to vertical
set(gca,'FontSize',18)
set(gcf,'WindowState','maximized')
%%
saveas(gcf,['../FiguresUsedinthePaper/',filename,'_GPR_Breiman'],'fig')
saveas(gcf,['../FiguresUsedinthePaper/',filename,'_GPR_Breiman'],'eps')
saveas(gcf,['../FiguresUsedinthePaper/',filename,'_GPR_Breiman'],'jpg')
saveas(gcf,['../FiguresUsedinthePaper/',filename,'_GPR_Breiman'],'png')

writematrix(AAGPR,'MoonResults.xlsx','FileType','spreadsheet','Sheet','GPR')

save(filename,'AANN','AASVM','AAGPR','STi')