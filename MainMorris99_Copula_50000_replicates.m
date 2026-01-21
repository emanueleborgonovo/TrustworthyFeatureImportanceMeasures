% Boston Housing Dataset
% Figures 10, 13, 22 Predictions After Permutation

close all;clearvars;clc
%%
Alreadytrained=1%input('Already Trained?; 1 if true, 0 if false ');
Rep=100%input('Number of Replicates ');
d=99;
labels = arrayfun(@(k) sprintf('X_{%d}', k), 1:d, 'UniformOutput', false);


%%
filename = matlab.desktop.editor.getActiveFilename;
[~, filename, ~] = fileparts(filename);
netname=strcat('trainednet',filename);
netnamemat=strcat(netname,'.mat')
% synthetic data generation
% important decreases with a(i)
%%
rng("default")
rho=0.5;
d=99;
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


%%

% Limit Range: if lr=1, limit the range (truncated normal), otherwise infinity
% tr is the value at which to truncate
%[n,d]=size(x);
%lr=1;
%tr=2;
%indx=zeros(size(x));
%if lr==1
%    aux=0;
%    for i=1:d
%        indx=find(or(x(:,i)>tr,x(:,i)<-tr));
%    x(indx,:)=[];
%    end
%end
% With this procedure, we reduce the sample size, but we can overcompensate
% sampling a bit more. For instance, N=15000 instead of N=10000.

mdl=@(x) Model99(x);
y=Model99(x);
VY=var(y)
[W,d,nn]=wassersi(x,y,15);



%[Si1,STi1]=ihssi(d,N,mdl,@(u)u)

[Si,STi]=windsi(Sigma,N,mdl,@(u)u)
%[Si,STi]=windsi(Sigma,N,mdl,@normcdf)

Wass1=W.W1;
Wass2=W.W22/(2*var(y));
Wass8=W.W8;

%%
figure
subplot(5,1,1)
bar(Wass1)
title('Wasserstein 1')
subplot(5,1,2)
bar(Wass2)
title('Wasserstein 2')
subplot(5,1,3)
bar(Wass8)
title('Wasserstein 8')
subplot(5,1,4)
bar(Si)
title('Si from windsi')
subplot(5,1,5)
bar(STi)
title('STi from windsi')
%%
figure
bar(STi)
title(strcat('Ground Truth; rho=',num2str(rho)))
xlabel('Input')
ylabel('\tau_i')
set(gca,'FontSize',24)
set(gcf,'WindowState','maximized')
saveas(gcf,'../FiguresUsedinthePaper\GroundTruth_30000_50','fig')
saveas(gcf,'../FiguresUsedinthePaper\GroundTruth_30000_50','eps')
%saveas(gcf,'../FiguresUsedinthePaper\GroundTruth_30000_0','.jpg')
saveas(gcf,'../FiguresUsedinthePaper\GroundTruth_30000_50','png')
%%
%data=readmatrix('Housing.xlsx');
%x=data(:,1:end-2);
n=N;
data=[x y];
%y=data(:,end-1);
% Generating Training Data
%rdp=randperm(n);
dataaux1=data;
xaux1=dataaux1(:,1:end-1);
yaux1=dataaux1(:,end);

pTrain=0.80;
ntrain=round(n*pTrain);
xtrain=dataaux1(1:ntrain,1:end-1);
xtest=dataaux1(ntrain+1:end,1:end-1);
ytrain=dataaux1(1:ntrain,end);
ytest=dataaux1(ntrain+1:end,end);
%xstd=(x - min(x(:)))./(max(x(:)) - min(x(:)));
xx=x';
tt = y';
%%%%%%%%%%%%%%%%%%%%%
%% Data visualization
figure
[c0,y1]=ksdensity(y);
plot(y1,c0,"Color",'b','LineWidth',2)
hold on
xlabel('Y','FontSize',18)
y05=prctile(y,5);
y95=prctile(y,95);
xline(y05,'-.b','5th Quantile','LineWidth',1);xline(y95,'-.b','95th Quantile','LineWidth',1)
mm=mean(y);
vv=var(y);
ylabel('Empirical Density','FontSize',24);
uu=strcat('E[Y]=',num2str(mean(y)));
uuu=strcat('V[Y]=',num2str(var(y)));
text(mean(y), 0.015,uu,'FontSize', 24);
text(mean(y), 0.01, uuu,'FontSize', 24);
set(gcf,'WindowState','maximized')
set(gca,'FontSize',24)
title('Target Empirical Distribution','FontSize',24);
saveas(gcf,'../FiguresUsedinthePaper\OutputDensity_30000_50','fig')
saveas(gcf,'../FiguresUsedinthePaper\OutputDensity_30000_50','eps')
%saveas(gcf,'../FiguresUsedinthePaper\OutputDensity_30000_50','.jpg')
saveas(gcf,'../FiguresUsedinthePaper\OutputDensity__30000_50','png')

%%%%%%%%%%%%%%%%%%%%%

%% Artificial Neural Network
if Alreadytrained==0
rng(4508);
trainFcn = 'trainlm';  % Levenberg-Marquardt backpropagation.
hiddenLayerSize = [20 20];
net = fitnet(hiddenLayerSize,trainFcn);
net.input.processFcns = {'removeconstantrows','mapminmax'};
net.output.processFcns = {'removeconstantrows','mapminmax'};
net.divideFcn = 'dividerand';  % Divide data randomly
net.divideMode = 'sample';  % Divide up every sample
net.divideParam.trainRatio = pTrain;
net.divideParam.valRatio = (1-pTrain);
net.divideParam.testRatio = 0/100;
net.performFcn = 'mse';  % Mean Squared Error
net.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
    'plotregression', 'plotfit'};

% Train the Network
[net,tr] = train(net,xx,tt);

% Test the Network
ynet = net(xx);
e = gsubtract(tt,ynet);
performance = perform(net,tt,ynet);

% Recalculate Training, Validation and Test Performance
trainTargets = tt .* tr.trainMask{1};
valTargets = tt .* tr.valMask{1};
%testTargets = tt .* tr.testMask{1};
trainPerformance = perform(net,trainTargets,ynet);
valPerformance = perform(net,valTargets,ynet);
%testPerformance = perform(net,testTargets,ynet);
%R2netnet=computeR2(testTargets,ynet);

mdl=@(x) net(x')';
RMSENN=sqrt(sum((y-ynet').^2)/length(y));
MADNN=sum(abs(y-ynet'))/length(ynet);
R2Net=1 - sum((y - ynet').^2)/sum((y - mean(y)).^2); %0.9128

ynettest=mdl(xtest);
R2Nettest=1-sum((ytest - ynettest).^2)/sum((ytest - mean(ytest)).^2); %0.8894

save(netnamemat, 'net');
else
    load(netnamemat,'net')
mdl=@(x) net(x')';
ynet = net(x');
RMSENN=sqrt(sum((y-ynet').^2)/length(y));
MADNN=sum(abs(y-ynet'))/length(ynet);
R2Net=1 - sum((y - ynet').^2)/sum((y - mean(y)).^2); %0.9128

ynettest = mdl(xtest);
R2Nettest = 1 - sum((ytest - ynettest).^2)/sum((ytest - mean(ytest)).^2); %0.8894
end

%% Neural Network
% Figure 10, TAX (X10), predictions after restriction 
% RF: Variable Importance over Replicates
namemdl='NN';
mdl=@(x) net(x')';
Rep=5
YtestKS=1;
indices=[2];
for jj=1:Rep
    jj
    if jj==Rep
        Yplot=1;
    else
        Yplot=0;
    end
[nuNNUnraux,tauprime,TestResultsUnrAux]=nuBrei(x,y,mdl,namemdl,Yplot,indices,YtestKS,'density');
nuNNUnr(jj,:)=nuNNUnraux;
NN_TestResultsUnr1(jj,:)=TestResultsUnrAux(1,:);
NN_TestResultsUnr2(jj,:)=TestResultsUnrAux(2,:);
NN_TestResultsUnr3(jj,:)=TestResultsUnrAux(3,:);
[nuNNGCMRaux,NN_tauprimeGCMR,TestResultsGCMRAux]=nuGCMR(x,y,mdl,namemdl,Yplot,indices,YtestKS,'density');
nuNNGCMR(jj,:)=nuNNGCMRaux;
NN_TestResultsGCMR1(jj,:)=TestResultsGCMRAux(1,:);
NN_TestResultsGCMR2(jj,:)=TestResultsGCMRAux(2,:);
NN_TestResultsGCMR3(jj,:)=TestResultsGCMRAux(3,:);
[nuNNKnockaux,tauprimerestrKnock,TestResultsGKnockAux]=fnuGKnock(x,y,mdl,namemdl,Yplot,indices,YtestKS,'density');
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
legend('Ground Truth','$\nu_j$', '$\nu_j^{GCMR}$', '$\nu_j^{GKnock}$','FontSize',28,'Interpreter','latex','location','NorthWest')
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


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Linear Regression
% Figure 13  ZN (X2), predictions after restriction 
rng(12345)
linreg=fitlm(xtrain,ytrain);%,'interactions');
ylinreg=predict(linreg,xtest);
Vlinreg=var(ylinreg);
mdl2=@(x) predict(linreg,x);
RMSElinreg=sqrt(sum((ytest- ylinreg).^2)/length(ytest));
MADlinreg=sum(abs(ytest-ylinreg))/length(ytest);
R2linreg = 1 - sum((ytest - ylinreg).^2)/sum((ytest - mean(ytest)).^2); %0.8825
namemdl='Linear Regression';
%[TauALElinreg,KALElinreg]=aleplot(xtest,mdl2,20);
%%
namemdl='LR';
mdl=@(x) net(x')';
YtestKS=1;
indices=[2];
for jj=1:Rep
    jj
    if jj==Rep
        Yplot=1;
    else
        Yplot=0;
    end
[nuLRUnraux,tauprime,TestResultsUnrAux]=nuBrei(x,y,mdl,namemdl,Yplot,indices,YtestKS,'density');
nuLRUnr(jj,:)=nuLRUnraux;
LR_TestResultsUnr1(jj,:)=TestResultsUnrAux(1,:);
LR_TestResultsUnr2(jj,:)=TestResultsUnrAux(2,:);
LR_TestResultsUnr3(jj,:)=TestResultsUnrAux(3,:);
[nuLRGCMRaux,LR_tauprimeGCMR,TestResultsGCMRAux]=nuGCMR(x,y,mdl,namemdl,Yplot,indices,YtestKS,'density');
nuLRGCMR(jj,:)=nuLRGCMRaux;
LR_TestResultsGCMR1(jj,:)=TestResultsGCMRAux(1,:);
LR_TestResultsGCMR2(jj,:)=TestResultsGCMRAux(2,:);
LR_TestResultsGCMR3(jj,:)=TestResultsGCMRAux(3,:);
[nuLRKnockaux,tauprimerestrKnock,TestResultsGKnockAux]=fnuGKnock(x,y,mdl,namemdl,Yplot,indices,YtestKS,'density');
nuLRKnock(jj,:)=nuLRKnockaux;
LR_TestResultsGKnock1(jj,:)=TestResultsGKnockAux(1,:);
LR_TestResultsGKnock2(jj,:)=TestResultsGKnockAux(2,:);
LR_TestResultsGKnock3(jj,:)=TestResultsGKnockAux(3,:);
waitbar(jj/Rep);
end


AALR=[STi;mean(nuLRUnr)/(2*VY); mean(nuLRGCMR)/(2*VY); mean(nuLRKnock)/(2*VY)];
figure
b = bar(AALR'); b(1).FaceColor = [0 0 0]; b(2).FaceColor = [0 0 1]; b(3).FaceColor = [1 0 0]; b(4).FaceColor=[0 1 0];
title('Linear Regression, $\nu_j$, $\nu_j^{GCMR}$ and $\nu_j^{GKnock}$','FontSize',30,'Interpreter','latex')
legend('Ground Truth','$\nu_j$', '$\nu_j^{GCMR}$', '$\nu_j^{GKnock}$','FontSize',28,'Interpreter','latex','location','NorthWest')
xticklabels(labels)
%set(gca,'TickLabelInterpreter','none')
xtickangle(90); % Set orientation to vertical
set(gca,'FontSize',18)
set(gcf,'WindowState','maximized')
%%
saveas(gcf,['../FiguresUsedinthePaper/',filename,'_LR_Breiman'],'fig')
saveas(gcf,['../FiguresUsedinthePaper/',filename,'_LR_Breiman'],'eps')
saveas(gcf,['../FiguresUsedinthePaper/',filename,'_LR_Breiman'],'jpg')
saveas(gcf,['../FiguresUsedinthePaper/',filename,'_LR_Breiman'],'png')


save(filename,'AANN','AALR','STi','-mat')

%% Performance Printing
disp('----- Neural Network ----')
fprintf('Coefficient of Determination NN: %f\n',R2Net);
fprintf('Mean Squared Error NN: %f\n',RMSENN);
fprintf('Mean Absolute Deviation NN: %f\n',MADNN);

disp('----- Linear Regression ----')
fprintf('Coefficient of Determination Linear Regression: %f\n',R2linreg);
fprintf('Mean Squared Error Random Forest: %f\n',RMSElinreg);
fprintf('Mean Absolute Deviation: %f\n',MADlinreg);



%% Ground Truth Comparisons
figure
AA=[STi; nuNetUnr/(2*VY); nuNetGCRM/(2*VY); nuKnock/(2*VY)];
BB=[tauprime; tauprimeGCRM;tauprimerestrKnock];
subplot(3,1,1)
b = bar(AA');%b(1).FaceColor = [0 0 1];b(2).FaceColor = [1 0 0];b(3).FaceColor=[0 1 0];
title('Neural Network')
%ylabel('Variable Importance')
legend('Ground Truth','$\nu_j$', '$\nu_j^{GCMR}$', '$\nu_j^{GKnock}$','Interpreter','latex','Location', 'northwest')
xticklabels({'X_{10}','X_{20}','X_{30}','X_{40}','X_{50}','X_{60}','X_{70}','X_{80}','X_{90}','X_{10}'})
set(gca,'FontSize',20)

subplot(3,1,2)
AA=[STi; nuLinRregUnr/(2*VY); nuLinRregGCRM/(2*VY); nuLinRregKnock/(2*VY)];
b = bar(AA');
title('Linear Regression')
%ylabel('Variable Importance')
legend('Ground Truth','$\nu_j$', '$\nu_j^{GCMR}$', '$\nu_j^{GKnock}$','Interpreter','latex','Location', 'northwest')
xticklabels({'X_{10}','X_{20}','X_{30}','X_{40}','X_{50}','X_{60}','X_{70}','X_{80}','X_{90}','X_{10}'})
set(gca,'FontSize',20)

set(gcf,'WindowState','maximized')
saveas(gcf,'../FiguresUsedinthePaper\Model99_All_50000_GroundTruth','fig')
saveas(gcf,'../FiguresUsedinthePaper\Model99_All_50000_GroundTruth','eps')
%saveas(gcf,'../FiguresUsedinthePaper\RF_Breiman_30000_0','.jpg')
saveas(gcf,'../FiguresUsedinthePaper\Model99_All_50000_GroundTruth','png')
