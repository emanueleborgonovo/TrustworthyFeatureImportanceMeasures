%% BP papar figure 9 and 10, c), d) permutation restricted
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

correxperiment = 1
rho = 0.9;

% correxperiment = 0
% rho = 0

%% RF : Permute, OOB, BP 
FIrfBPnuM = nan(rep,k);% permute rf nu
FIrfBPtauM = nan(rep,k);% permute rf tau

for j = 1:rep
    p = haltonset(k,'Skip',seeds(j),'Leap',1e2);
    p = scramble(p,'RR2');
    u=net(p,n+1);
    uu=u(2:end,:);
    X = uu;
    if correxperiment == 1
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
    [nuRestr,tauprimeresrt]=BreimanRestrictedLM(X,Y,mdl,'model', 0);
    runtime = toc;
    fprintf(['RF Rep ', num2str(j), ' BP FI finished, timeuse = ',num2str(runtime),'sec. \n'])

    FIrfBPnuM(j,:) = nuRestr;
    FIrfBPtauM(j,:) = tauprimeresrt;

end

%% LM: Permute, BP 
FIlmBPnuM = nan(rep,k);% permute lm nu
FIlmBPtauM = nan(rep,k);% permute lm tau

for j = 1:rep
    p = haltonset(k,'Skip',seeds(j),'Leap',1e2);
    p = scramble(p,'RR2');
    u=net(p,n+1);
    uu=u(2:end,:);
    X = uu;
    if correxperiment == 1
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
    [nuRestr,tauprimeresrt]=BreimanRestrictedLM(X,Y,mdl,'model', 0);
    runtime = toc;
    fprintf(['LM Rep ', num2str(j), ' BP FI finished, timeuse = ',num2str(runtime),'sec. \n'])
    FIlmBPnuM(j,:) = nuRestr;
    FIlmBPtauM(j,:) = tauprimeresrt;

end


%% NN: Permute, BP 
FInnBPnuM = nan(rep,k);% permute nn nu
FInnBPtauM = nan(rep,k);% permute nn tau

for j = 1:rep
    p = haltonset(k,'Skip',seeds(j),'Leap',1e2);
    p = scramble(p,'RR2');
    u=net(p,n+1);
    uu=u(2:end,:);
    X = uu;
    if correxperiment == 1
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
    [nuRestr,tauprimeresrt]=BreimanRestrictedLM(X,Y,mdl,'model', 0);
    runtime = toc;
    fprintf(['NN Rep ', num2str(j), ' BP FI finished, timeuse = ',num2str(runtime),'sec. \n'])
    FInnBPnuM(j,:) = nuRestr;
    FInnBPtauM(j,:) = tauprimeresrt;

end



clearvars uu u p epsilon
%% plot
clearvars
%load('result00_uniform_lmrfnn_GRIM_lm.mat'); rho = 0;
load('result09_uniform_lmrfnn_GRIM_lm.mat'); rho = 0.9
betas = [1,1,1,1,1,0,0.5,0.8,1.2,1.5];
n = 2000; %2000
k = 10;
esigma = 0.1;
p = haltonset(k,'Skip',1234,'Leap',1e2);
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
%% convert to ranking
FIrfBPnuMrank = Value2Rank(FIrfBPnuM);
FIrfBPtauMrank = Value2Rank(FIrfBPtauM);
FIlmBPnuMrank = Value2Rank(FIlmBPnuM);
FIlmBPtauMrank = Value2Rank(FIlmBPtauM);
FInnBPnuMrank = Value2Rank(FInnBPnuM);
FInnBPtauMrank = Value2Rank(FInnBPtauM);
%% ranking plot - restricted
if rho == 0.9
    totalS = [0.0166   0.0167   0.0878   0.0877   0.0878        0   0.0219   0.0562   0.1264   0.1975]
    ranksanl = Value2Rank(totalS);%[3,3, 7,7 7, 1, 4,5,9,10];
else
    % rho = 0
    totalS = [1,1,1,1,1,0,0.5,0.8,1.2,1.5].^2* var(Y)*(1-0.9^2);
    ranksanl = Value2Rank(totalS);%[6,6,6,6,6,1, 2,3,9,10];
end
figure
plot(1:k, ranksanl, '-ok','LineWidth',2); hold on
plot(1:k, mean(FIlmBPnuMrank), '-^r','LineWidth',2)
plot(1:k, mean(FIrfBPnuMrank), '-+g','LineWidth',2)
plot(1:k, mean(FInnBPnuMrank), '-xb','LineWidth',2)
grid on
xlabel('feature')
ylabel('importance rank')
title(['rho = ', num2str(rho),', GRIM lm'])
legend({'analy','LM','RF','NN'}, Location="southeast")
% set(gcf, 'PaperPosition', [0 0 10 7]); %[left bottom width height]
% set(gcf, 'PaperSize', [10 7]);
% saveas(gcf, ['Hoocker_nu_rho0'], 'pdf')

%% Figure 10, c) d)
figure
plot(1:k, ranksanl, '-ok','LineWidth',2); hold on
plot(1:k, mean(FIlmBPtauMrank), '-^r','LineWidth',2)
plot(1:k, mean(FIrfBPtauMrank), '-+g','LineWidth',2)
plot(1:k, mean(FInnBPtauMrank), '-xb','LineWidth',2)
grid on
xlabel('feature')
ylabel('importance rank')
title(['rho = ', num2str(rho),', GRIM lm'])
legend({'analy','LM','RF','NN'}, Location="southeast")

% set(gcf, 'PaperPosition', [0 0 10 7]); %[left bottom width height]
% set(gcf, 'PaperSize', [10 7]);
% saveas(gcf, ['Hoocker_tau_rho0'], 'pdf')

%% save results
if rho == 0.9
    save("result09_uniform_lmrfnn_GRIM_lm.mat", 'FIrfBPnuM','FIrfBPtauM','FIlmBPnuM','FIlmBPtauM','FInnBPnuM','FInnBPtauM')
else
     save("result00_uniform_lmrfnn_GRIM_lm.mat", 'FIrfBPnuM','FIrfBPtauM','FIlmBPnuM','FIlmBPtauM','FInnBPnuM','FInnBPtauM')
end

%% Importance plot - restricted
if rho == 0.9
    totalS = [0.0166   0.0167   0.0878   0.0877   0.0878        0   0.0219   0.0562   0.1264   0.1975]
    ranksanl = Value2Rank(totalS)%[3,3, 7,7 7, 1, 4,5,9,10];
else
    % rho = 0
    totalS = [1,1,1,1,1,0,0.5,0.8,1.2,1.5].^2* var(Y)*(1-0.9^2);
    ranksanl = Value2Rank(totalS)%[6,6,6,6,6,1, 2,3,9,10];
end
figure
plot(1:k, totalS, '-ok','LineWidth',2); hold on
plot(1:k, mean(FIlmBPnuM), '-^r','LineWidth',2)
plot(1:k, mean(FIrfBPnuM), '-+g','LineWidth',2)
plot(1:k, mean(FInnBPnuM), '-xb','LineWidth',2)
grid on
xlabel('feature')
ylabel('feature importance')
title(['rho = ', num2str(rho),', GRIM lm'])
legend({'analy','LM','RF','NN'}, Location="eastoutside")


