%% Hooker case
% Machine learning model: linear regression, random forest, neurual network 
% Importance measures: Brieman's unrestriced nu, nu^GCMR, nu^GKnock
clearvars; clc; close all
%% Experimental settings
rep = 50; rng(7777)
seeds = round(rand(rep,1)*10000);

betas = [1,1,1,1,1,0,0.5,0.8,1.2,1.5];
n = 2000; 
k = 10;
esigma = 0.1;
rho = 0.9;
%% Model: Linear Regression
FIlmnuPM = nan(rep,k);
FIlmGCMRnuM = nan(rep,k);
FIlmGKnocknuM = nan(rep,k);

namemdl='Linear Regression';% plot name
Yplot=0;
indices = []; % plot index
hWaitbar = waitbar(0, 'Starting LR iterations...'); % Create the waitbar
for j = 1:rep
    % Generate Data
    p = haltonset(k,'Skip',seeds(j),'Leap',1e2);
    p = scramble(p,'RR2');
    u=net(p,n+1);
    uu=u(2:end,:);
    X = uu;
    if rho == 0.9
    X(:,1:2) = copularnd('gaussian',rho,n);
    end
    epsilon = normrnd(0,esigma,n,1);
    Y =  betas * X';
    Y = Y' + epsilon;

    % build LM
    rng(seeds(j)); % For reproducibility
    mdl_lm = fitlm(X,Y);
    mdl = @(X)predict(mdl_lm,X);   

    % Unrestricted permutation importance   
    [nuNetUnr,tauprime]=BreimanUnr(X,Y,mdl,namemdl,Yplot,indices);
    FIlmnuPM(j,:) = nuNetUnr;
    % GCMR
    [nuNetGCRM,tauprimeGCRM]=nuGCMR(X,Y,mdl,namemdl,Yplot,indices);
    FIlmGCMRnuM(j,:)=nuNetGCRM;
    % GKnockoffs
    [nuKnock,tauprimerestrKnock]=fnuGKnock(X,Y,mdl,namemdl,Yplot,indices);
    FIlmGKnocknuM(j,:)=nuKnock;

    waitbar(j /rep, hWaitbar, sprintf('Progress: %d%%', round(100 * j / rep)));
end
close(hWaitbar);
%% Model Random Forest
FIrfnuPM = nan(rep,k);
FIrfGCMRnuM = nan(rep,k);
FIrfGKnocknuM = nan(rep,k);

namemdl='Random Forest';% plot name
Yplot=0;
indices = []; % plot index
hWaitbar = waitbar(0, 'Starting RF iterations...'); % Create the waitbar
for j = 1:rep
    % Generate Data
    p = haltonset(k,'Skip',seeds(j),'Leap',1e2);
    p = scramble(p,'RR2');
    u=net(p,n+1);
    uu=u(2:end,:);
    X = uu;
    if rho == 0.9
    X(:,1:2) = copularnd('gaussian',rho,n);
    end
    epsilon = normrnd(0,esigma,n,1);
    Y =  betas * X';
    Y = Y' + epsilon;

    % Build RF
    rng(seeds(j)); % For reproducibility
    Mdl_rf = fitrensemble(X,Y,'Method','Bag');
    mdl = @(X)predict(Mdl_rf,X);
   
    % Unrestricted permutation importance   
    [nuNetUnr,tauprime]=BreimanUnr(X,Y,mdl,namemdl,Yplot,indices);
    FIrfnuPM(j,:) = nuNetUnr;
    % GCMR
    [nuNetGCRM,tauprimeGCRM]=nuGCMR(X,Y,mdl,namemdl,Yplot,indices);
    FIrfGCMRnuM(j,:)=nuNetGCRM;
    % GKnockoffs
    [nuKnock,tauprimerestrKnock]=fnuGKnock(X,Y,mdl,namemdl,Yplot,indices);
    FIrfGKnocknuM(j,:)=nuKnock;

    waitbar(j /rep, hWaitbar, sprintf('Progress: %d%%', round(100 * j / rep)));
end
close(hWaitbar);
%% Model: Neural Networks
FInnnuPM = nan(rep,k);
FInnGCMRnuM = nan(rep,k);
FInnGKnocknuM = nan(rep,k);

namemdl='Neural Networks';% plot name
Yplot=0;
indices = []; % plot index
hWaitbar = waitbar(0, 'Starting NN iterations...'); % Create the waitbar
for j = 1:rep
    % Generate Data
    p = haltonset(k,'Skip',seeds(j),'Leap',1e2);
    p = scramble(p,'RR2');
    u=net(p,n+1);
    uu=u(2:end,:);
    X = uu;
    if rho == 0.9
    X(:,1:2) = copularnd('gaussian',rho,n);
    end
    epsilon = normrnd(0,esigma,n,1);
    Y =  betas * X';
    Y = Y' + epsilon;

    % Build NN
    rng(seeds(j)); % For reproducibility
    nnnet = fitnet(20);
    nnnet.trainParam.showWindow = false;
    mdl_net = train(nnnet, X', Y');
    mdl = @(X) mdl_net(X')';  

    % Unrestricted permutation importance   
    [nuNetUnr,tauprime]=BreimanUnr(X,Y,mdl,namemdl,Yplot,indices);
    FInnnuPM(j,:) = nuNetUnr;
    % GCMR
    [nuNetGCRM,tauprimeGCRM]=nuGCMR(X,Y,mdl,namemdl,Yplot,indices);
    FInnGCMRnuM(j,:)=nuNetGCRM;
    % GKnockoffs
    [nuKnock,tauprimerestrKnock]=fnuGKnock(X,Y,mdl,namemdl,Yplot,indices);
    FInnGKnocknuM(j,:)=nuKnock;

    waitbar(j /rep, hWaitbar, sprintf('Progress: %d%%', round(100 * j / rep)));
end
close(hWaitbar);

clearvars uu u p epsilon
%% Define Analytical values
totalS = [0.0166   0.0167   0.0878   0.0877   0.0878        0   0.0219   0.0562   0.1264   0.1975];
ranksanl = Value2Rank(totalS);
%% Figure 7 a)
FIrfnuPMrank = Value2Rank(FIrfnuPM);
FIlmnuPMrank = Value2Rank(FIlmnuPM);
FInnnuPMrank = Value2Rank(FInnnuPM);

figure
subplot(1,2,1)
plot(1:k, 2*totalS, '-ok','LineWidth',2,'MarkerSize',10,'MarkerFaceColor','auto'); hold on
plot(1:k, mean(FIlmnuPM), '-^r','LineWidth',2,'MarkerSize',10,'MarkerFaceColor','r')
plot(1:k, mean(FIrfnuPM), '-+g','LineWidth',2,'MarkerSize',15,'MarkerFaceColor','g')
plot(1:k, mean(FInnnuPM), '-xb','LineWidth',2,'MarkerSize',15,'MarkerFaceColor','b')
grid on
xlabel('$X_j$','FontSize',24,'interpreter','latex')
ylabel('$\hat{\nu}_j$','FontSize',28,'interpreter','latex')
title(['\textbf{$\rho$ = ', num2str(rho),', unrestricted}'],'FontSize',24,'interpreter','latex')
legend({'Analytical','Linear Model','Random Forest','Neural Network'},'Location','northwest','FontSize',14)
ax = gca; ax.FontSize = 20;

subplot(1,2,2)
plot(1:k, ranksanl, '-ok','LineWidth',2,'MarkerSize',10,'MarkerFaceColor','auto'); hold on
plot(1:k, mean(FIlmnuPMrank), '-^r','LineWidth',2,'MarkerSize',10,'MarkerFaceColor','r')
plot(1:k, mean(FIrfnuPMrank), '-+g','LineWidth',2,'MarkerSize',15,'MarkerFaceColor','g')
plot(1:k, mean(FInnnuPMrank), '-xb','LineWidth',2,'MarkerSize',15,'MarkerFaceColor','b')
grid on
xlabel('$X_j$','FontSize',24,'interpreter','latex')
ylabel('Importance Rank','FontSize',24)
title(['\textbf{$\rho$ = ', num2str(rho),', unrestricted}'],'FontSize',24,'interpreter','latex')
legend({'Analytical','Linear Model','Random Forest','Neural Network'},'Location','north','FontSize',14)
ax = gca; ax.FontSize = 20;ylim([1,11])

%set(gcf, 'PaperPosition', [-0.5 0 18 7]/1.3); %[left bottom width height]
%set(gcf, 'PaperSize', [17 7]/1.3);
%saveas(gcf, 'main_Fig9_LMRFNN_unrestricted', 'pdf') 

%% Figure 7 b) GCMR
% convert to ranking
FIrfBPnuMrank = Value2Rank(FIrfGCMRnuM);
FIlmBPnuMrank = Value2Rank(FIlmGCMRnuM);
FInnBPnuMrank = Value2Rank(FInnGCMRnuM);

figure
subplot(1,2,1)
plot(1:k, 2*totalS, '-ok','LineWidth',2,'MarkerSize',10,'MarkerFaceColor','auto'); hold on
plot(1:k, mean(FIlmGCMRnuM), '-^r','LineWidth',2,'MarkerSize',10,'MarkerFaceColor','r')
plot(1:k, mean(FIrfGCMRnuM), '-+g','LineWidth',2,'MarkerSize',15,'MarkerFaceColor','g')
plot(1:k, mean(FInnGCMRnuM), '-xb','LineWidth',2,'MarkerSize',15,'MarkerFaceColor','b')
grid on
xlabel('$X_j$','FontSize',24,'interpreter','latex')
ylabel('$\hat{\nu}_j^{GCMR}$','FontSize',28,'interpreter','latex')
title(['\textbf{$\rho$ = ', num2str(rho),', GCMR}'],'FontSize',24,'interpreter','latex')
legend({'Analytical','Linear Model','Random Forest','Neural Network'},'Location','northwest','FontSize',14)
ax = gca; ax.FontSize = 20;

subplot(1,2,2)
plot(1:k, ranksanl, '-ok','LineWidth',2,'MarkerSize',10,'MarkerFaceColor','auto'); hold on
plot(1:k, mean(FIlmBPnuMrank), '-^r','LineWidth',2,'MarkerSize',10,'MarkerFaceColor','r')
plot(1:k, mean(FIrfBPnuMrank), '-+g','LineWidth',2,'MarkerSize',15,'MarkerFaceColor','g')
plot(1:k, mean(FInnBPnuMrank), '-xb','LineWidth',2,'MarkerSize',15,'MarkerFaceColor','b')
grid on
xlabel('$X_j$','FontSize',24,'interpreter','latex')
ylabel('Importance Rank','FontSize',24)
title(['\textbf{$\rho$ = ', num2str(rho),', GCMR}'],'FontSize',24,'interpreter','latex')
legend({'Analytical','Linear Model','Random Forest','Neural Network'},'Location','north','FontSize',14)
ax = gca; ax.FontSize = 20; ylim([1,11])
% set(gcf, 'PaperPosition', [-0.5 0 18 7]/1.3); %[left bottom width height]
% set(gcf, 'PaperSize', [17 7]/1.3);
% saveas(gcf, 'main_Fig9_LMRFNN_GRIM_lm', 'pdf') 
%% Figure 7 c) GKnockoffs
FIrfBPnuMrank = Value2Rank(FIrfGKnocknuM);
FIlmBPnuMrank = Value2Rank(FIlmGKnocknuM);
FInnBPnuMrank = Value2Rank(FInnGKnocknuM);

figure
subplot(1,2,1)
plot(1:k, 2*totalS, '-ok','LineWidth',2,'MarkerSize',10,'MarkerFaceColor','auto'); hold on
plot(1:k, mean(FIlmGKnocknuM), '-^r','LineWidth',2,'MarkerSize',10,'MarkerFaceColor','r')
plot(1:k, mean(FIrfGKnocknuM), '-+g','LineWidth',2,'MarkerSize',15,'MarkerFaceColor','g')
plot(1:k, mean(FInnGKnocknuM), '-xb','LineWidth',2,'MarkerSize',15,'MarkerFaceColor','b')
grid on
xlabel('$X_j$','FontSize',24,'interpreter','latex')
ylabel('$\hat{\nu}_j^{GKnock}$','FontSize',28,'interpreter','latex')
title(['\textbf{$\rho$ = ', num2str(rho),', Gknock}'],'FontSize',24,'interpreter','latex')
legend({'Analytical','Linear Model','Random Forest','Neural Network'},'Location','northwest','FontSize',14)
ax = gca; ax.FontSize = 20;

subplot(1,2,2)
plot(1:k, ranksanl, '-ok','LineWidth',2,'MarkerSize',10,'MarkerFaceColor','auto'); hold on
plot(1:k, mean(FIlmBPnuMrank), '-^r','LineWidth',2,'MarkerSize',10,'MarkerFaceColor','r')
plot(1:k, mean(FIrfBPnuMrank), '-+g','LineWidth',2,'MarkerSize',15,'MarkerFaceColor','g')
plot(1:k, mean(FInnBPnuMrank), '-xb','LineWidth',2,'MarkerSize',15,'MarkerFaceColor','b')
grid on
xlabel('$X_j$','FontSize',24,'interpreter','latex')
ylabel('Importance Rank','FontSize',24)
title(['\textbf{$\rho$ = ', num2str(rho),', Gknock}'],'FontSize',24,'interpreter','latex')
legend({'Analytical','Linear Model','Random Forest','Neural Network'},'Location','north','FontSize',14)
ax = gca; ax.FontSize = 20;ylim([1,11])
% set(gcf, 'PaperPosition', [-0.5 0 18 7]/1.3); %[left bottom width height]
% set(gcf, 'PaperSize', [17 7]/1.3);
% saveas(gcf, 'main_Fig9_LMRFNN_knockoff', 'pdf') 

