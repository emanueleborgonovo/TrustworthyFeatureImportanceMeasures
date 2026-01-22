% BP papar figure 9, a) b) unrestricted
clearvars; clc
%close all;
r2 = @(y,ypred) 1 - sum((y - ypred).^2)/sum((y - mean(y)).^2);
%% 
rep = 50; %50
rng(7777)
seeds = round(rand(rep,1)*10000);


betas = [1,1,1,1,1,0,0.5,0.8,1.2,1.5];
n = 2000; %2000
k = 10;
esigma = 0.1;

rho = 0.9;
% rho = 0
%% RF : Permute, OOB, BP 
FIoobM = nan(rep,k);
FIrfnuPM = nan(rep,k);% permute rf
FIrftauPM = nan(rep,k);% permute rf
for j = 1:rep
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


    % build RF
    rng(seeds(j)); % For reproducibility
    % t = templateTree('NumVariablesToSample','all',...
    % 'PredictorSelection','interaction-curvature','Surrogate','on');
    % Mdl_rf = fitrensemble(X,Y,'Method','Bag','NumLearningCycles',200, ...
    % 'Learners',t);
    Mdl_rf = fitrensemble(X,Y,'Method','Bag');
    % oob importance
    FIoobM(j,:) = oobPermutedPredictorImportance(Mdl_rf);

    mdl = @(X)predict(Mdl_rf,X);
   
    % permutation importance   
    tic
    [FInu, FItau]=PermutationBasedFeatureImportance(X,Y,mdl);
    FIrfnuPM(j,:) = FInu;
    FIrftauPM(j,:) = FItau;
    runtime = toc;
    fprintf(['RF Rep ', num2str(j), ' permutation FI finished, timeuse = ',num2str(runtime),'sec. \n'])
end

%% LM: Permute, BP 
FIlmnuPM = nan(rep,k);% permute lm
FIlmtauPM = nan(rep,k);% permute lm

for j = 1:rep
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
    %mdl_lm.Coefficients;

    mdl = @(X)predict(mdl_lm,X);
   
    % permutation importance   
    tic
    [FInu, FItau]=PermutationBasedFeatureImportance(X,Y,mdl);
    FIlmnuPM(j,:) = FInu;
    FIlmtauPM(j,:) = FItau;
    runtime = toc;
    fprintf(['LM Rep ', num2str(j), ' permutation FI finished, timeuse = ',num2str(runtime),'sec. \n'])

end

%% NN: Permute, BP 
FInnnuPM = nan(rep,k);% permute nn
FInntauPM = nan(rep,k);% permute nn

for j = 1:rep
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
    % permutation importance   
    tic
    [FInu, FItau]=PermutationBasedFeatureImportance(X,Y,mdl);
    FInnnuPM(j,:) = FInu;
    FInntauPM(j,:) = FItau;
    runtime = toc;
    fprintf(['NN Rep ', num2str(j), ' permutation FI finished, timeuse = ',num2str(runtime),'sec. \n'])

end

clearvars uu u p epsilon

%% save results
if rho == 0.9
    save("result09_uniform_lmrfnn.mat",'FIoobM',  'FIrfnuPM','FIlmnuPM','FInnnuPM',...
        'FIrftauPM','FIlmtauPM','FInntauPM')
else
    save("result00_uniform_lmrfnn.mat",'FIoobM',  'FIrfnuPM','FIlmnuPM','FInnnuPM',...
        'FIrftauPM','FIlmtauPM','FInntauPM')
end


%% plot
clearvars;close all
load('result09_uniform_lmrfnn.mat')
rho = 0.9;k=10;
totalS = [0.0166   0.0167   0.0878   0.0877   0.0878        0   0.0219   0.0562   0.1264   0.1975];
ranksanl = Value2Rank(totalS);%[3,3, 7,7 7, 1, 4,5,9,10];

% convert to ranking
FIoobMrank = Value2Rank(FIoobM);
FIrfnuPMrank = Value2Rank(FIrfnuPM);
FIrftauPMrank = Value2Rank(FIrftauPM);
FIlmnuPMrank = Value2Rank(FIlmnuPM);
FIlmtauPMrank = Value2Rank(FIlmtauPM);
FInnnuPMrank = Value2Rank(FInnnuPM);
FInntauPMrank = Value2Rank(FInntauPM);

figure
subplot(1,2,1)
plot(1:k, 2*totalS, '-ok','LineWidth',2); hold on
plot(1:k, mean(FIlmnuPM), '-^r','LineWidth',2)
plot(1:k, mean(FIrfnuPM), '-+g','LineWidth',2)
plot(1:k, mean(FInnnuPM), '-xb','LineWidth',2)
grid on
xlabel('$X_j$','FontSize',24,'interpreter','latex')
ylabel('$\hat{\nu}_j$','FontSize',28,'interpreter','latex')
title(['rho = ', num2str(rho),', unrestricted'],'FontSize',18)
legend({'Analytical','Linear Model','Random Forest','Neural Network'},'Location','northwest','FontSize',14)

subplot(1,2,2)
plot(1:k, ranksanl, '-ok','LineWidth',2); hold on
plot(1:k, mean(FIlmnuPMrank), '-^r','LineWidth',2)
plot(1:k, mean(FIrfnuPMrank), '-+g','LineWidth',2)
plot(1:k, mean(FInnnuPMrank), '-xb','LineWidth',2)
grid on
xlabel('$X_j$','FontSize',24,'interpreter','latex')
ylabel('Importance Rank','FontSize',24)
title(['rho = ', num2str(rho),', unrestricted'],'FontSize',18)
legend({'Analytical','Linear Model','Random Forest','Neural Network'},'Location','northwest','FontSize',14)

%%
% rho = 0;k=10;
% totalS = [0.0913    0.0913    0.0913    0.0913    0.0913         0    0.0228    0.0584    0.1315    0.2054];% nu = 2*totalS
%0.1535    0.1535    0.1535    0.1535    0.1535         0    0.0384    0.0982    0.2211    0.3454

