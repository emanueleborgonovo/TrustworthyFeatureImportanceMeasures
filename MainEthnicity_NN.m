clear all
close all
clc

% A=readtable('Ethnicityschort.csv');
% n = 100; % number of obs to calculate restricted importance

A=readtable('EthnicIA_TrainingSet.csv');
n = 1300000%100000; %1000000;
%% encode the variable
[C,idxC,idxC1]=unique(A.race); % {'Asian','Black','Hispanic','White'   }
A.race = idxC1;
A.race = categorical(A.race);
%% identify inputs and output
y =A.race;
%figure,histogram(y);
%
rng(10,'twister')
part = cvpartition(y,'HoldOut',0.2);
istrain = training(part);
istest = test(part);
ytrain = y(istrain);
ytest = y(istest);
Xtrain = A(istrain,:);
Xtest = A(istest,:);
Xtrain(:,{'race'}) = [];
Xtest(:,{'race'}) = [];
featurenames=A.Properties.VariableNames(2:end);%Xtest.Properties.VariableNames;
Xtest = table2array(Xtest);
Xtrain = table2array(Xtrain);
tabulate(ytrain)
tabulate(ytest)
%% train a neural network
% tic
% rng(12345)
% % adjust NN topology here
% options = statset('UseParallel',true);
% Mdl = fitcnet(Xtrain,ytrain,"LayerSizes",[7 21 7]); % change here for NN topology
% toc % 37mins

load('EthnicityResult_NN_n1300000.mat')
%% model evaluation - in sample
tic
[ytrainnet,score_train] = predict(Mdl,Xtrain);
toc
figure
confusionchart(ytrain,ytrainnet,"Normalization","row-normalized");
title('train performance')
Cmatrix_train = confusionmat(ytrain,ytrainnet');
Result_common_train = multiclass_metrics_common(Cmatrix_train)      
    %   Precision: 0.8544
    %      Recall: 0.7934
    %    Accuracy: 0.8826
    % Specificity: 0.9329
    %     F1score: 0.8185
% %Roc curve
rocObj_train= rocmetrics(ytrain,score_train,Mdl.ClassNames);
rocObj_train.AUC %    0.9193    0.9363    0.9709    0.9442
figure
plot(rocObj_train)
title('ROC Curve, Train')
%% model evaluation - out of sample
tic
[ytestnet,score_test] = predict(Mdl,Xtest);
toc
figure
confusionchart(ytest,ytestnet,"Normalization","row-normalized");
title('test performance')
Cmatrix_test = confusionmat(ytest,ytestnet');
Result_common_test = multiclass_metrics_common(Cmatrix_test)      
   % Precision: 0.8547
   %       Recall: 0.7938
   %     Accuracy: 0.8826
   %  Specificity: 0.9329
   %      F1score: 0.8189
% %Roc curve
rocObj_test = rocmetrics(ytest,score_test,Mdl.ClassNames);
rocObj_test.AUC %    0.9192    0.9364    0.9708    0.9442
figure
plot(rocObj_test)
title('ROC Curve, Test')

%% Unrestricted

mdl = @(x) predict(Mdl,x) ;

Yplot=0;namemdl = 'ANN Unrestricted';
indices=[1:size(Xtest,2)];
[nuUnrNetm,nuUnrNetallm]=BreimanUnrClassification(Xtest(1:n,:),ytest(1:n),mdl,namemdl,Yplot,indices);
fprintf(['Unrestricted class finished. \n'])


%% GCMR
mdl = @(x) predict(Mdl,x) ;
Yplot=0;namemdl = 'ANN GCMR';
[nuNetRestrm,nuNetRestrallm]=BreimanRestrictedClassification(Xtest(1:n,:),ytest(1:n),mdl,namemdl,Yplot,indices);
fprintf(['GCMR restricted finished. \n'])
% 
% save(['EthnicityResult_NN_n',num2str(n),'.mat'],'Mdl','featurenames',...
%      ...   'Result_common_test','Result_common_train','rocObj_test','rocObj_train',...
%     'nuUnrNetm','nuUnrNetallm',...
%     'nuNetRestrm',"nuNetRestrallm")

%% knockoff
mdl = @(x) predict(Mdl,x) ;
Yplot=0;namemdl = 'ANN knockoff';
indices=[1:size(Xtest,2)];
tic
[nuRestknockoffm,nuRestknockoffallm]=BreimanKnockoffsClassification(Xtest(1:n,:),ytest(1:n),mdl,namemdl,Yplot,indices);
runtime = toc;
fprintf(['Knockoff restricted finished, timeuse = ',num2str(runtime),'sec. \n'])

%
save(['EthnicityResult_NN_n',num2str(n),'.mat'],'Mdl','featurenames',...
    ... 'Result_common_test','Result_common_train','rocObj_test','rocObj_train',...
    'nuUnrNetm','nuUnrNetallm',...
    'nuNetRestrm',"nuNetRestrallm",...
    "nuRestknockoffm","nuRestknockoffallm")

%% plot
close all
%load('EthnicityResult_NN_n1300000_short.mat')

figure
colororder = [0.8500 0.3250 0.0980;
    0 0.4470 0.7410;
    0.9290 0.6940 0.1250;
    0.4660 0.6740 0.1880];
fs = 16;
classnames = {'Asian','Black','Hispanic','White'   };

% Unrestricted
t = tiledlayout(1,4);
plotdata = flip(nuUnrNetm,2);
for i = 1:size(nuUnrNetm, 1)
    nexttile
    barh(categorical(1:size(nuUnrNetm, 2)),plotdata(i,:),'FaceColor',colororder(i,:))
    yaxisproperties= get(gca, 'YAxis');
    yaxisproperties.TickLabelInterpreter = 'none';
    title('\nu_j, unrestricted')
    xlabel(classnames(i))
    xlim([min(plotdata(i,:))-0.01,max(plotdata(i,:))*1.1])
    % Set YTick and YTickLabel properties
    if i == 1
        ylabel('Features')
        set(gca, 'YTickLabel', flip(featurenames), 'FontSize',fs)
    else
        set(gca, 'YTick', [], 'YTickLabel', [], 'FontSize',fs)
    end

end
t.TileSpacing = 'compact';
t.Padding = 'compact';

% GCMR
figure
t = tiledlayout(1,4);
plotdata = flip(nuNetRestrm,2);
for i = 1:size(plotdata, 1)
    nexttile
    barh(categorical(1:size(plotdata, 2)),plotdata(i,:),'FaceColor',colororder(i,:))
    yaxisproperties= get(gca, 'YAxis');
    yaxisproperties.TickLabelInterpreter = 'none';
    title('\nu^{GCMR}_j, restricted')
    xlabel(classnames(i))
    xlim([min(plotdata(i,:))-0.01,max(plotdata(i,:))*1.1])
    % Set YTick and YTickLabel properties
    if i == 1
        ylabel('Features')
        set(gca, 'YTickLabel', flip(featurenames), 'FontSize',fs)
    else
        set(gca, 'YTick', [], 'YTickLabel', [], 'FontSize',fs)
    end

end
t.TileSpacing = 'compact';
t.Padding = 'compact';

%Knockoffs
figure
t = tiledlayout(1,4);
plotdata = flip(nuRestknockoffm,2);
for i = 1:size(plotdata, 1)
    nexttile
    barh(categorical(1:size(plotdata, 2)),plotdata(i,:),'FaceColor',colororder(i,:))
    yaxisproperties= get(gca, 'YAxis');
    yaxisproperties.TickLabelInterpreter = 'none';
    title('\nu^{GKnock}_j, restricted')
    xlabel(classnames(i))
    xlim([min(plotdata(i,:))-0.01,max(plotdata(i,:))*1.1])
    % Set YTick and YTickLabel properties
    if i == 1
        ylabel('Features')
        set(gca, 'YTickLabel', flip(featurenames), 'FontSize',fs)
    else
        set(gca, 'YTick', [], 'YTickLabel', [], 'FontSize',fs)
    end

end
t.TileSpacing = 'compact';
t.Padding = 'compact';



