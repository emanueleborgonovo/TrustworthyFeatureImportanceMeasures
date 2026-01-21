%  Friedman 10D example
% % Figures 10, 13, 22 Predictions After Permutation

close all;clearvars;clc

% synthetic data generation
% important decreases with a(i)
rng("default")
rho=0.5;
d=10;
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

filename=strcat('Friedman10_',num2str(N),'_',num2str(rho),'.xlsx')

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

mdl=@(x) friedman10(x);
y=friedman10(x);
VY=var(y)
[W,d,nn]=wassersi(x,y,15);

%[Si1,STi1]=ihssi(d,N,mdl,@(u)u)
[Si,STi]=windsi(Sigma,N,@friedman10,@(u)u)

%[Si,STi]=windsi(Sigma,N,mdl,@normcdf)

Wass1=W.W1;
Wass2=W.W22/(2*var(y));
Wass8=W.W8;

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

figure
bar(STi)
title(strcat('Ground Truth; rho=',num2str(rho)))
xlabel('Input')
ylabel('\tau_i')
set(gca,'FontSize',24)
set(gcf,'WindowState','maximized')
saveas(gcf,'../FiguresUsedinthePaper\Friedman10_50000_50_GroundTruth','fig')
saveas(gcf,'../FiguresUsedinthePaper\Friedman10_50000_50_GroundTruth','eps')
%saveas(gcf,'../FiguresUsedinthePaper\Friedman10_GroundTruth_50000_0','.jpg')
saveas(gcf,'../FiguresUsedinthePaper\Friedman10_50000_50_GroundTruth','png')

writematrix(STi,filename)
%%

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
saveas(gcf,'../FiguresUsedinthePaper\Friedman10_OutputDensity_50000_50','fig')
saveas(gcf,'../FiguresUsedinthePaper\Friedman10_OutputDensity_50000_50','eps')
%saveas(gcf,'../FiguresUsedinthePaper\Friedman10_OutputDensity_50000_50','.jpg')
saveas(gcf,'../FiguresUsedinthePaper\Friedman10_OutputDensity__50000_50','png')

%%%%%%%%%%%%%%%%%%%%%

%% Artificial Neural Network
rng(4508);
trainFcn = 'trainlm';  % Levenberg-Marquardt backpropagation.
hiddenLayerSize = [20 20 20 20];
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

ynettest = mdl(xtest);
R2Nettest = 1 - sum((ytest - ynettest).^2)/sum((ytest - mean(ytest)).^2); %0.8894

%save('Friedman10_50000Net.mat','net')
%%
load('Friedman10_50000Net.mat','net')
%[TauALEnet,KALEnet]=aleplot(x,mdl, 20);
%% Neural Network
% Figure 10, TAX (X10), predictions after restriction 
namemdl='Neural Network';
Yplot=1;
indices=[];

rng(12345)
tic
[nuNetUnr,tauprime]=BreimanUnr(x,y,mdl,namemdl,Yplot,indices);
tBR_NN=toc
tic
[nuNetGCRM,tauprimeGCRM]=nuGCMR(x,y,mdl,namemdl,Yplot,indices);
t_BR_GCMR_NN=toc
tic
[nuKnock,tauprimerestrKnock]=fnuGKnock(x,y,mdl,namemdl,Yplot,indices);
t_BR_GKnock_NN=toc

%% One sample of Figure 9 result
AANN=[nuNetUnr/(2*VY); nuNetGCRM/(2*VY); nuKnock/(2*VY)];
BB=[tauprime; tauprimeGCRM;tauprimerestrKnock];
figure
b = bar(AANN');%b(1).FaceColor = [0 0 1];b(2).FaceColor = [1 0 0];b(3).FaceColor=[0 1 0];
title('Neural Network, $\nu_j$, $\nu_j^{GCMR}$ and $\nu_j^{GKnock}$','FontSize',30,'Interpreter','latex')
legend('$\nu_j$', '$\nu_j^{GCMR}$', '$\nu_j^{GKnock}$','FontSize',28,'Interpreter','latex')
xticklabels({'X_{1}','X_{2}','X_{3}','X_{4}','X_{5}','X_{6}','X_{7}','X_{8}','X_{9}','X_{10}'})
set(gca,'FontSize',24)
set(gcf,'WindowState','maximized')
saveas(gcf,'../FiguresUsedinthePaper\Friedman10_NN_Breiman_50000_50','fig')
saveas(gcf,'../FiguresUsedinthePaper\Friedman10_NN_Breiman_50000_50','eps')
%saveas(gcf,'../FiguresUsedinthePaper\Friedman10_NN_Breiman_50000_50','jpg')
saveas(gcf,'../FiguresUsedinthePaper\Friedman10_NN_Breiman_50000_50','png')
writematrix(AANN,filename,'Sheet','NeuralNetwork')



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Linear Regression
% Figure 13  ZN (X2), predictions after restriction 
rng(12345)
linreg=fitlm(xtrain,ytrain,'interactions');
ylinreg=predict(linreg,xtest);
Vlinreg=var(ylinreg);
mdl2=@(x) predict(linreg,x);
RMSElinreg=sqrt(sum((ytest- ylinreg).^2)/length(ytest));
MADlinreg=sum(abs(ytest-ylinreg))/length(ytest);
R2linreg = 1 - sum((ytest - ylinreg).^2)/sum((ytest - mean(ytest)).^2); %0.8825
namemdl='Linear Regression';
%[TauALElinreg,KALElinreg]=aleplot(xtest,mdl2,20);
%%
rng(12345)
Yplot=1;
indices=[];

tic
[nuLinRregUnr,tauprimeLinRreg]=BreimanUnr(x,y,mdl2,namemdl,Yplot,indices);
t_BR_LinReg=toc
tic
[nuLinRregGCRM,tauprimeLinRregGCRM]=nuGCMR(x,y,mdl2,namemdl,Yplot,indices);
t_BR_GCMR_LinReg=toc
tic
[nuLinRregKnock,tauprimeLinRregKnock]=fnuGKnock(x,y,mdl2,namemdl,Yplot,indices);
t_BR_GKnock_LinReg=toc
tic

%% One sample of Figure 12 result
AA=[nuLinRregUnr; nuLinRregGCRM; nuLinRregKnock];
BB=[tauprimeLinRreg; tauprimeLinRregGCRM;tauprimeLinRregKnock];
figure
b = bar(AA');b(1).FaceColor = [0 0 1];b(2).FaceColor = [1 0 0];b(3).FaceColor=[0 1 0];
title('Linear Regression, $\nu_j$, $\nu_j^{GCMR}$ and $\nu_j^{GKnock}$','FontSize',30,'Interpreter','latex')
legend('$\nu_j$', '$\nu_j^{GCMR}$', '$\nu_j^{GKnock}$','FontSize',28,'Interpreter','latex')
xticklabels({'X_{1}','X_{2}','X_{3}','X_{4}','X_{5}','X_{6}','X_{7}','X_{8}','X_{9}','X_{10}'})
set(gca,'FontSize',24)
set(gcf,'WindowState','maximized')
saveas(gcf,'../FiguresUsedinthePaper\Friedman10_LinReg_Breiman_50000_50','fig')
saveas(gcf,'../FiguresUsedinthePaper\Friedman10_LinReg_Breiman_50000_50','eps')
saveas(gcf,'../FiguresUsedinthePaper\Friedman10_LinReg_Breiman_50000_50','jpg')
saveas(gcf,'../FiguresUsedinthePaper\Friedman10_LinReg_Breiman_50000_50','png')

AALR=[nuLinRregUnr/(2*VY); nuLinRregGCRM/(2*VY); nuLinRregKnock/(2*VY)];
writematrix(AALR,filename,'Sheet','Regression')
percErrAALR=abs(STi-AALR)./STi;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Random Forest
% Figure 22
rng(12345)
regtree=fitrensemble(xtrain,ytrain);
namemdl='Random Forest';
yregtree=predict(regtree,xtest);
R2regtree=1 - sum((ytest - yregtree).^2)/sum((ytest - mean(ytest)).^2);
MADRegTree = mean(abs(ytest-yregtree));
RMSEregtree=sqrt(mean((ytest-yregtree).^2));

mdl3=@(x) predict(regtree,x);

%[TauALEregtree,KALERegTree]=aleplot(x,mdl3,20);
Yplot=1;
indices=[];
rng(12345)
tic
[nuRFUnr,tauprimeRF]=BreimanUnr(x,y,mdl3,namemdl,Yplot,indices);
t_BR_RF=toc
tic
[nuRFGCRM,tauprimeRFGCRM]=nuGCMR(x,y,mdl3,namemdl,Yplot,indices);
t_GCMR_RF=toc
tic
[nuRFKnock,tauprimeRFKnock]=fnuGKnock(x,y,mdl3,namemdl,Yplot,indices);
t_GKnock_RF=toc


%% One sample of Figure 21 result
AA=[nuRFUnr/(2*VY); nuRFGCRM/(2*VY); nuRFKnock/(2*VY)];
BB=[tauprimeRF; tauprimeRFGCRM;tauprimeRFKnock];
figure
b = bar(AA');b(1).FaceColor = [0 0 1];b(2).FaceColor = [1 0 0];b(3).FaceColor=[0 1 0]
title('Random Forest, $\nu_j$, $\nu_j^{GCMR}$ and $\nu_j^{GKnock}$','FontSize',30,'Interpreter','latex')
legend('$\nu_j$', '$\nu_j^{GCMR}$', '$\nu_j^{GKnock}$','FontSize',28,'Interpreter','latex')
xticklabels({'X_{1}','X_{2}','X_{3}','X_{4}','X_{5}','X_{6}','X_{7}','X_{8}','X_{9}','X_{10}'})
set(gca,'FontSize',24)
set(gcf,'WindowState','maximized')
saveas(gcf,'../FiguresUsedinthePaper\Friedman10_RF_Breiman_50000_0','fig')
saveas(gcf,'../FiguresUsedinthePaper\Friedman10_RF_Breiman_50000_0','eps')
%saveas(gcf,'../FiguresUsedinthePaper\Friedman10_RF_Breiman_50000_0','.jpg')
saveas(gcf,'../FiguresUsedinthePaper\Friedman10_RF_Breiman_50000_0','png')

AARF=[nuRFUnr/(2*VY); nuRFGCRM/(2*VY); nuRFKnock/(2*VY)];
writematrix(AARF,filename,'Sheet','RF')
percErrRF=abs(STi-AARF)./STi;



%% Performance Printing
disp('----- Neural Network ----')
fprintf('Coefficient of Determination NN: %f\n',R2Net);
fprintf('Mean Squared Error NN: %f\n',RMSENN);
fprintf('Mean Absolute Deviation NN: %f\n',MADNN);

disp('----- Linear Regression ----')
fprintf('Coefficient of Determination Linear Regression: %f\n',R2linreg);
fprintf('Mean Squared Error Random Forest: %f\n',RMSElinreg);
fprintf('Mean Absolute Deviation: %f\n',MADlinreg);


disp('----- Random Forest ----')
fprintf('Coefficient of Determination Random Forest: %f\n',R2regtree);
fprintf('Mean Squared Error Random Forest: %f\n',RMSEregtree);
fprintf('Mean Absolute Deviation: %f\n',MADRegTree);

%% Ground Truth Comparisons
figure
AA=[STi; nuNetUnr/(2*VY); nuNetGCRM/(2*VY); nuKnock/(2*VY)];
BB=[tauprime; tauprimeGCRM;tauprimerestrKnock];
subplot(3,1,1)
b = bar(AA');b(1).FaceColor ='k';b(2).FaceColor = [0 0 1];b(3).FaceColor = [1 0 0];b(4).FaceColor=[0 1 0];
title('Neural Network')
%ylabel('Variable Importance')
legend('Ground Truth','$\nu_j$', '$\nu_j^{GCMR}$', '$\nu_j^{GKnock}$','Interpreter','latex','Location', 'northeast')
xticklabels({'X_{1}','X_{2}','X_{3}','X_{4}','X_{5}','X_{6}','X_{7}','X_{8}','X_{9}','X_{10}'})
set(gca,'FontSize',20)

subplot(3,1,2)
AA=[STi; nuLinRregUnr/(2*VY); nuLinRregGCRM/(2*VY); nuLinRregKnock/(2*VY)];
b = bar(AA');b(1).FaceColor ='k';b(2).FaceColor = [0 0 1];b(3).FaceColor = [1 0 0];b(4).FaceColor=[0 1 0];
title('Linear Regression')
%ylabel('Variable Importance')
legend('Ground Truth','$\nu_j$', '$\nu_j^{GCMR}$', '$\nu_j^{GKnock}$','Interpreter','latex','Location', 'northeast')
xticklabels({'X_{1}','X_{2}','X_{3}','X_{4}','X_{5}','X_{6}','X_{7}','X_{8}','X_{9}','X_{10}'})
set(gca,'FontSize',20)

subplot(3,1,3)
AA=[STi; nuRFUnr/(2*VY); nuRFGCRM/(2*VY); nuRFKnock/(2*VY)];
BB=[tauprimeRF; tauprimeRFGCRM;tauprimeRFKnock];
b = bar(AA');b(1).FaceColor ='k';b(2).FaceColor = [0 0 1];b(3).FaceColor = [1 0 0];b(4).FaceColor=[0 1 0];
title('Random Forest')
legend('Ground Truth','$\nu_j$', '$\nu_j^{GCMR}$', '$\nu_j^{GKnock}$','Interpreter','latex','Location', 'northeast')
%ylabel('Variable Importance')
xticklabels({'X_{1}','X_{2}','X_{3}','X_{4}','X_{5}','X_{6}','X_{7}','X_{8}','X_{9}','X_{10}'})
set(gca,'FontSize',20)
set(gcf,'WindowState','maximized')
saveas(gcf,'../FiguresUsedinthePaper\Friedman_All_50000_GroundTruth','fig')
saveas(gcf,'../FiguresUsedinthePaper\Friedman_All_50000_GroundTruth','eps')
%saveas(gcf,'../FiguresUsedinthePaper\RF_Breiman_50000_0','.jpg')
saveas(gcf,'../FiguresUsedinthePaper\Friedman_All_50000_GroundTruth','png')

%% Percentage Errors Comparison
figure
subplot(3,1,1)
percErrNN=abs(STi-AANN)./STi;
b = bar(percErrNN');b(1).FaceColor = [0 0 1];b(2).FaceColor = [1 0 0];b(3).FaceColor=[0 1 0];
title('Neural Network')
%ylabel('Variable Importance')
legend('$\nu_j$', '$\nu_j^{GCMR}$', '$\nu_j^{GKnock}$','Interpreter','latex','Location', 'northeast')
xticklabels({'X_{1}','X_{2}','X_{3}','X_{4}','X_{5}','X_{6}','X_{7}','X_{8}','X_{9}','X_{10}'})
set(gca,'FontSize',20)

subplot(3,1,2)
percErrLR=abs(STi-AALR)./STi;
b = bar(percErrLR');b(1).FaceColor = [0 0 1];b(2).FaceColor = [1 0 0];b(3).FaceColor=[0 1 0];
title('Linear Regression')
%ylabel('Variable Importance')
legend('Ground Truth','$\nu_j$', '$\nu_j^{GCMR}$', '$\nu_j^{GKnock}$','Interpreter','latex','Location', 'northeast')
xticklabels({'X_{1}','X_{2}','X_{3}','X_{4}','X_{5}','X_{6}','X_{7}','X_{8}','X_{9}','X_{10}'})
set(gca,'FontSize',20)

subplot(3,1,3)
percErrRF=abs(STi-AARF)./STi;
b = bar(percErrRF');b(1).FaceColor = [0 0 1];b(2).FaceColor = [1 0 0];b(3).FaceColor=[0 1 0];
title('Random Forest')
legend('Ground Truth','$\nu_j$', '$\nu_j^{GCMR}$', '$\nu_j^{GKnock}$','Interpreter','latex','Location', 'northeast')
%ylabel('Variable Importance')
xticklabels({'X_{1}','X_{2}','X_{3}','X_{4}','X_{5}','X_{6}','X_{7}','X_{8}','X_{9}','X_{10}'})
set(gca,'FontSize',20)
set(gcf,'WindowState','maximized')
saveas(gcf,'../FiguresUsedinthePaper\Friedman_All_50000_PercError','fig')
saveas(gcf,'../FiguresUsedinthePaper\Friedman_All_50000_PercError','eps')
%saveas(gcf,'../FiguresUsedinthePaper\RF_Breiman_50000_0','.jpg')
saveas(gcf,'../FiguresUsedinthePaper\Friedman_All_50000_PercError','png')

writematrix(percErrNN,filename,'Sheet','ErrorsNN')
writematrix(percErrLR,filename,'Sheet','ErrorsRegression')
writematrix(percErrRF,filename,'Sheet','ErrorsRF')