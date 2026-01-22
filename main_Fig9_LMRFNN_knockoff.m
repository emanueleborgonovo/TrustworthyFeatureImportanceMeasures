%% BP papar figure 9 and 10 permutation restricted
clearvars; clc
%close all;
r2 = @(y,ypred) 1 - sum((y - ypred).^2)/sum((y - mean(y)).^2);
%% 
rep = 50; %50
rng(7777)
seeds = round(rand(rep,1)*10000);


betas = [1,1,1,1,1,0,0.5,0.8,1.2,1.5];
n = 2000; %10000; %2000
k = 10;
esigma = 0.1;

rho = 0.9;

%correxperiment = 0
%rho = 0

%% RF : Permute, OOB, BP 
FIrfBPnuM = nan(rep,k);% permute rf nu
FIrfBPtauM = nan(rep,k);% permute rf tau
gen=1 %(Random);

for j = 1:rep
if gen==0
    p = haltonset(k,'Skip',seeds(j),'Leap',1e2);
    p = scramble(p,'RR2');
    u=net(p,n+1);
    uu=u(2:end,:);
    X = uu;
else
    X=rand(n,k);
end
    if rho == 0.9
    X(:,1:2) = copularnd('gaussian',rho,n);
    end
    epsilon = normrnd(0,esigma,n,1);
    Y =  betas * X';
    Y = Y' + epsilon;


    % build RF
    rng(seeds(j)); % For reproducibility
    % t = templateTree('NumVariablesToSample','all',...
    % 'PredictorSelection','interaction-curvature','Surrogate','on');
    % Mdl_rf = fitrensemble(X,Y,'Method','Bag','NumLearningCycles',200, ...
    % 'Learners',t);
    Mdl_rf = fitrensemble(X,Y,'Method','Bag'); % default setting not work?
    mdl = @(X)predict(Mdl_rf,X);

    tic
    [nuRestr,tauprimeresrt]=BreimanKnockoffs(X,Y,mdl,'model', 0);
    runtime = toc;
    fprintf(['RF Rep ', num2str(j), ' BP FI finished, timeuse = ',num2str(runtime),'sec. \n'])

    FIrfBPnuM(j,:) = nuRestr;
    FIrfBPtauM(j,:) = tauprimeresrt;

end

%% LM: Permute, BP 
FIlmBPnuM = nan(rep,k);% permute lm nu
FIlmBPtauM = nan(rep,k);% permute lm tau

for j = 1:rep
    if gen==0
    p = haltonset(k,'Skip',seeds(j),'Leap',1e2);
    p = scramble(p,'RR2');
    u=net(p,n+1);
    uu=u(2:end,:);
    X = uu;
else
    X=rand(n,k);
end
    if rho == 0.9
    X(:,1:2) = copularnd('gaussian',rho,n);
    end
    epsilon = normrnd(0,esigma,n,1);
    Y =  betas * X';
    Y = Y' + epsilon;

    % build LM
    rng(seeds(j)); % For reproducibility
    mdl_lm = fitlm(X,Y);
    %mdl_lm.Coefficients;

    mdl = @(X)predict(mdl_lm,X);

    tic
    [nuRestr,tauprimeresrt]=BreimanKnockoffs(X,Y,mdl,'model', 0);
    runtime = toc;
   % [nuRestrf,tauprimeresrtf]=BreimanKnockoffs_false(X,Y,mdl,'model', 0);
    % [nuRestrfrf,tauprimeresrtfrf]=BreimanRestrictedRF(X,Y,mdl,'model', 0);
    fprintf(['LM Rep ', num2str(j), ' BP FI finished, timeuse = ',num2str(runtime),'sec. \n'])
    FIlmBPnuM(j,:) = nuRestr;
    FIlmBPtauM(j,:) = tauprimeresrt;

end


%% NN: Permute, BP 
FInnBPnuM = nan(rep,k);% permute nn nu
FInnBPtauM = nan(rep,k);% permute nn tau

for j = 1:rep
    if gen==0
    p = haltonset(k,'Skip',seeds(j),'Leap',1e2);
    p = scramble(p,'RR2');
    u=net(p,n+1);
    uu=u(2:end,:);
    X = uu;
else
    X=rand(n,k);
end
    if rho == 0.9
    X(:,1:2) = copularnd('gaussian',rho,n);
    end
    epsilon = normrnd(0,esigma,n,1);
    Y =  betas * X';
    Y = Y' + epsilon;
    % 
    % % build nn
    % rng(seeds(j)); % For reproducibility
    % % The network has one hidden layer with ten neurons.
    % nnnet = feedforwardnet(20); 
    % nnnet.trainParam.showWindow = false;
    % % define transformation function
    % nnnet.layers{1}.transferFcn = 'radbas'; % 'poslin', 'logsig'X, 'tansig'X
    % [mdl_net,~] = train(nnnet,X',Y');
    % 
    rng(seeds(j)); % For reproducibility
    nnnet = fitnet(20);
    % Set the random seed for reproducibility (optional)
    nnnet.trainParam.showWindow = false;
    % Train the neural network
    mdl_net = train(nnnet, X', Y');
    % 
    % 
    mdl = @(X) mdl_net(X')';  
    fprintf(['R2 = ',num2str(r2(Y,mdl(X))) ,'\n'])
    tic
    [nuRestr,tauprimeresrt]=BreimanKnockoffs(X,Y,mdl,'model', 0);
    runtime = toc;
    fprintf(['NN Rep ', num2str(j), ' BP FI finished, timeuse = ',num2str(runtime),'sec. \n'])
    FInnBPnuM(j,:) = nuRestr;
    FInnBPtauM(j,:) = tauprimeresrt;

end



clearvars uu u p epsilon

%% save results
if rho == 0.9
    save("result09_uniform_lmrfnn_knockoff.mat", 'FIrfBPnuM','FIrfBPtauM','FIlmBPnuM','FIlmBPtauM','FInnBPnuM','FInnBPtauM')
else
     save("result00_uniform_lmrfnn_knockoff.mat", 'FIrfBPnuM','FIrfBPtauM','FIlmBPnuM','FIlmBPtauM','FInnBPnuM','FInnBPtauM')
end

%%
clearvars;close all
load('result09_uniform_lmrfnn_knockoff.mat')
k=10; rho = 0.9;

totalS = [0.0166   0.0167   0.0878   0.0877   0.0878        0   0.0219   0.0562   0.1264   0.1975]
ranksanl = Value2Rank(totalS)%[3,3, 7,7 7, 1, 4,5,9,10];
% convert to ranking
FIrfBPnuMrank = Value2Rank(FIrfBPnuM);
FIrfBPtauMrank = Value2Rank(FIrfBPtauM);
FIlmBPnuMrank = Value2Rank(FIlmBPnuM);
FIlmBPtauMrank = Value2Rank(FIlmBPtauM);
FInnBPnuMrank = Value2Rank(FInnBPnuM);
FInnBPtauMrank = Value2Rank(FInnBPtauM);
%%
figure
subplot(1,2,1)
plot(1:k, 2*totalS, '-ok','LineWidth',2); hold on
plot(1:k, mean(FIlmBPnuM), '-^r','LineWidth',2)
plot(1:k, mean(FIrfBPnuM), '-+g','LineWidth',2)
plot(1:k, mean(FInnBPnuM), '-xb','LineWidth',2)
grid on
xlabel('$X_j$','FontSize',24,'interpreter','latex')
ylabel('$\hat{\nu}_j^{GKnock}$','FontSize',28,'interpreter','latex')
title(['rho = ', num2str(rho),', Gknock'],'FontSize',18)
legend({'Analytical','Linear Model','Random Forest','Neural Network'},'Location','northwest','FontSize',14)

subplot(1,2,2)
plot(1:k, ranksanl, '-ok','LineWidth',2); hold on
plot(1:k, mean(FIlmBPnuMrank), '-^r','LineWidth',2)
plot(1:k, mean(FIrfBPnuMrank), '-+g','LineWidth',2)
plot(1:k, mean(FInnBPnuMrank), '-xb','LineWidth',2)
grid on
xlabel('$X_j$','FontSize',24,'interpreter','latex')
ylabel('Importance Rank','FontSize',24)
title(['rho = ', num2str(rho),', Gknock'],'FontSize',18)
legend({'Analytical','Linear Model','Random Forest','Neural Network'},'Location','northwest','FontSize',14)


