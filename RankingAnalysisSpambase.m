% Example: list of vectors stored in a cell array
clc
close all
clear all
load('main_spambase_replicates_App_Ensamble_NN_SVM_discr.mat')
AA=who
%{'AABT'                  }
%    {'AANN'                  }
%    {'AASVM'                 }

%% Reduction ratio analysis
ratios=AABT(1,:)./AABT(2,:);


%%
top1=10;
top2=10;

% Write out the top feature for each method
vectAll=[AABT(2,:); AANN(2,:); AASVM(2,:); Wass2];
[svAll,siAll]=sort(vectAll,2,'descend');

method_names = {'Random Forest', 'NN', 'SVM','Wass2'};

fprintf('\n Top feature for each method:\n');
for i = 1:size(vectAll,1)
    fprintf('Method %s: Feature %d\n', method_names{i}, siAll(i,1));
end

% New addition: Show top 5 features for each method
fprintf('\nTop %d features for each method:\n',10);
for i = 1:size(vectAll,1)
    fprintf('Method %s: Features %s\n', method_names{i}, mat2str(siAll(i,1:10)));
end

%%
vectors = [AABT(2,:);AANN(2,:)];
[sv,si]=sort(vectors,2,'descend');
aux=0;
for i = 1:top1
   a=any(si(1,i)==si(2,1:top2));
aux=aux+a;
end
disp('Random Forest and Artificial NN ')
fprintf('\n Number top %d for Random Forest appearing in the top %d of Artificial NN: %d \n',top1,top2,aux);
%
vectors = [AABT(2,:);AASVM(2,:)];
[sv,si]=sort(vectors,2,'descend');
aux=0;
for i = 1:top1
   a=any(si(1,i)==si(2,1:top2));
aux=aux+a;
end
disp('Random Forest and SVM')
fprintf('\n Number top %d for Random Forest appearing in the top %d of SVM: %d \n',top1,top2,aux);

vectors = [AANN(2,:);AASVM(2,:)];
[sv,si]=sort(vectors,2,'descend');
aux=0;
for i = 1:top1
   a=any(si(1,i)==si(2,1:top2));
aux=aux+a;
end
disp('Artificial NN and SVM')
fprintf('\n Number top %d for Artificial NN appearing in the top %d of SVM: %d \n',top1,top2,aux);

%
vectors = [AABT(2,:);Wass2];
[sv,si]=sort(vectors,2,'descend');
aux=0;
for i = 1:top1
   a=any(si(1,i)==si(2,1:top2));
aux=aux+a;
end
disp('Random Forest and Artificial NN ')
fprintf('\n Number top %d for Random Forest appearing in the top %d of Wass2: %d \n',top1,top2,aux);


%%
names={'BT','NN','SVM','Wass2'};
table(corr([AABT(2,:);AANN(2,:);AASVM(2,:);Wass2]','type','spearman'),'rownames',names,'VariableNames',{'Pearson'}) %AASVM(2,:);
table(savagescorenewmatrix([AABT(2,:);AANN(2,:);AASVM(2,:);Wass2]'),'rownames',names,'VariableNames',{'Savage'}) %AASVM(2,:);

%% Statistical Comparison
figure
%subplot(2,1,1)
%b = bar(VShap);  b(1).FaceColor = [0 0 1]; 
%title('Cohort Variance-Based Shapley')

%legend('$\nu_j$', '$\nu_j^{GCMR}$', '$\nu_j^{GKnock}$','FontSize',28,'Interpreter','latex')
%ticks=[1:13];
%xticks(ticks)
%xticklabels(labels)
%ylabel('VShap')
%legend('Ground Truth','$\nu_j$', '$\nu_j^{GCMR}$', '$\nu_j^{GKnock}$','FontSize',28,'Interpreter','latex','Location','Northwest')
%%xtickangle(45)
%set(gca,'TickLabelInterpreter','none')
%xtickangle(90); % Set orientation to vertical
set(gca,'FontSize',24)
set(gcf,'WindowState','maximized')

%subplot(2,1,2)
b = bar(Wass2');  b(1).FaceColor = [0 0 1]; 
title('Wasserstein Correlation')
ylabel('$\iota^{W_2^2}$','Interpreter','latex')
%legend('$\nu_j$', '$\nu_j^{GCMR}$', '$\nu_j^{GKnock}$','FontSize',28,'Interpreter','latex')
%xticklabels(labels)
%set(gca,'TickLabelInterpreter','none')
%xtickangle(90); % Set orientation to vertical
set(gca,'FontSize',24)
set(gcf,'WindowState','maximized')

filename = matlab.desktop.editor.getActiveFilename;
[~, filename, ~] = fileparts(filename);

saveas(gcf,['../FiguresUsedinthePaper/',filename,'_Stat_Compar'],'fig')
saveas(gcf,['../FiguresUsedinthePaper/',filename,'_Stat_Compar'],'eps')
saveas(gcf,['../FiguresUsedinthePaper/',filename,'_Stat_Compar'],'jpg')
saveas(gcf,['../FiguresUsedinthePaper/',filename,'_Stat_Compar'],'png')

%%
AAA=[AABT;AANN;AASVM];

figure

subplot(3,1,1)
b = bar(AABT');  b(1).FaceColor = [0 0 1]; b(2).FaceColor = [1 0 0]; b(3).FaceColor=[0 1 0];
title('Random Forest')
%xlabel('Feature, $X_i$','Interpreter','latex')
%legend('$\nu_j$', '$\nu_j^{GCMR}$', '$\nu_j^{GKnock}$','FontSize',28,'Interpreter','latex')
%xticklabels(labels)
%set(gca,'TickLabelInterpreter','none')
%xtickangle(90); % Set orientation to vertical
set(gca,'FontSize',24)
set(gcf,'WindowState','maximized')


subplot(3,1,2)
b = bar(AANN');  b(1).FaceColor = [0 0 1]; b(2).FaceColor = [1 0 0]; b(3).FaceColor=[0 1 0];
title('Artificial NN')
%xlabel('Feature, $X_i$','Interpreter','latex')

%legend('$\nu_j$', '$\nu_j^{GCMR}$', '$\nu_j^{GKnock}$','FontSize',28,'Interpreter','latex')
%ticks=[1:13];
%xticks(ticks)
%xticklabels(labels)
legend('$\nu_j$', '$\nu_j^{GCMR}$', '$\nu_j^{GKnock}$','FontSize',28,'Interpreter','latex','Location','Northwest')
%%xtickangle(45)
%set(gca,'TickLabelInterpreter','none')
%xtickangle(90); % Set orientation to vertical
set(gca,'FontSize',24)
set(gcf,'WindowState','maximized')

subplot(3,1,3)
b = bar(AASVM');  b(1).FaceColor = [0 0 1]; b(2).FaceColor = [1 0 0]; b(3).FaceColor=[0 1 0];
title('SVM')
%xticklabels(labels)
%set(gca,'TickLabelInterpreter','none')
%xtickangle(90); % Set orientation to vertical
xlabel('Feature, $X_i$','Interpreter','latex')
set(gca,'FontSize',24)
set(gcf,'WindowState','maximized')

saveas(gcf,['../FiguresUsedinthePaper/',filename,'_All'],'fig')
saveas(gcf,['../FiguresUsedinthePaper/',filename,'_All'],'eps')
saveas(gcf,['../FiguresUsedinthePaper/',filename,'_All'],'jpg')
saveas(gcf,['../FiguresUsedinthePaper/',filename,'_All'],'png')

%% Analysis of Least Important Inputs
%%
top1=10;
top2=10;

% Write out the top feature for each method
vectAll=[AABT(2,:); AANN(2,:); AASVM(2,:); Wass2];
[svAll,siAll]=sort(vectAll,2,'ascend');

method_names = {'Random Forest', 'NN', 'SVM','Wass2'};

fprintf('\n Least feature for each method:\n');
for i = 1:size(vectAll,1)
    fprintf('Method %s: Feature %d\n', method_names{i}, siAll(i,1));
end

% New addition: Show top 5 features for each method
fprintf('\nLast %d features for each method:\n',10);
for i = 1:size(vectAll,1)
    fprintf('Method %s: Features %s\n', method_names{i}, mat2str(siAll(i,1:10)));
end

%%
vectors = [AABT(2,:);AANN(2,:)];
[sv,si]=sort(vectors,2,'ascend');
aux=0;
for i = 1:top1
   a=any(si(1,i)==si(2,1:top2));
aux=aux+a;
end
disp('Random Forest and Artificial NN ')
fprintf('\n Number Last %d for Random Forest appearing in the Last %d of Artificial NN: %d \n',top1,top2,aux);
%
vectors = [AABT(2,:);AASVM(2,:)];
[sv,si]=sort(vectors,2,'ascend');
aux=0;
for i = 1:top1
   a=any(si(1,i)==si(2,1:top2));
aux=aux+a;
end
disp('Random Forest and SVM')
fprintf('\n Number Last %d for Random Forest appearing in the Last %d of SVM: %d \n',top1,top2,aux);

vectors = [AANN(2,:);AASVM(2,:)];
[sv,si]=sort(vectors,2,'ascend');
aux=0;
for i = 1:top1
   a=any(si(1,i)==si(2,1:top2));
aux=aux+a;
end
disp('Artificial NN and SVM')
fprintf('\n Number Last %d for Artificial NN appearing in the Last %d of SVM: %d \n',top1,top2,aux);

%
vectors = [AABT(2,:);Wass2];
[sv,si]=sort(vectors,2,'ascend');
aux=0;
for i = 1:top1
   a=any(si(1,i)==si(2,1:top2));
aux=aux+a;
end
disp('Random Forest and Artificial NN ')
fprintf('\n Number Last %d for Random Forest appearing in the Last %d of Wass2: %d \n',top1,top2,aux);

