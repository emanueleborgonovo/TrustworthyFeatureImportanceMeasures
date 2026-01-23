clc
load('Moon_replicates_50000_app.mat')
filename = matlab.desktop.editor.getActiveFilename;
[~, filename, ~] = fileparts(filename);
d=20;
labels = arrayfun(@(k) sprintf('X_{%d}', k), 1:d, 'UniformOutput', false);
%%
AAA=[AANN;AASVM;AAGPR];
Aperc=[abs(AAA-STi)./STi];
Apercdisp=[Aperc([2:4,6:8,10:12],:)]
[aa,bb]=sort(AANN(2,:),'Descend');
%bb(1:7)
%%
figure
ndisplayed=5;
indx=bb(1:ndisplayed)
% --- NN(not stacked) ---
subplot(1,3,1)
data = Aperc(2:4,indx)';
b = bar(data);  % grouped bars
b(1).FaceColor = [0 0 1];  % Blue
b(2).FaceColor = [1 0 0];  % Red
b(3).FaceColor = [0 1 0];  % Green
title('Artificial NN')
ylim([0 2]);
legend('$\nu_j$','$\nu_j^{GCMR}$','$\nu_j^{GKnock}$','Interpreter','Latex','Location','northwest')
xticklabels(labels(indx))
xtickangle(0)
set(gca,'FontSize',24)

% Add values on top of bars using corresponding bar color
[nGroups, nBars] = size(data);
groupWidth = min(0.8, nBars/(nBars + 1.5));
for i = 1:nBars
    barColor = b(i).FaceColor;
    for j = [1:ndisplayed]
        x = j - groupWidth/2 + (2*i-1) * groupWidth / (2*nBars);
        yVal = data(j,i);
        if yVal > 0
            percentText = sprintf('%.1f%%', 100 * yVal); 
            if i==2
                aau=0.06
            else
                aau=0.01
            end
            text(x, yVal + aau,percentText, ...
                'HorizontalAlignment', 'center', ...
                'VerticalAlignment', 'bottom', ...
                'FontSize', 14, 'Color', barColor, 'FontWeight', 'bold');
        end
    end
end

% --- SVM (not stacked) ---
subplot(1,3,2)
data = Aperc(6:8,indx)';
b = bar(data);  % grouped bars
b(1).FaceColor = [0 0 1];  % Blue
b(2).FaceColor = [1 0 0];  % Red
b(3).FaceColor = [0 1 0];  % Green
ylim([0 2]);
title('SVM')
%legend('$\nu_j$','$\nu_j^{GCMR}$','$\nu_j^{GKnock}$','Interpreter','Latex','Location','northwest')
xticklabels(labels(indx))
xtickangle(0)
set(gca,'FontSize',24)

% Add values on top of bars using corresponding bar color
[nGroups, nBars] = size(data);
groupWidth = min(0.8, nBars/(nBars + 1.5));
for i = 1:nBars
    barColor = b(i).FaceColor;
    for j = [1:ndisplayed]
        x = j - groupWidth/2 + (2*i-1) * groupWidth / (2*nBars);
        yVal = data(j,i);
        if yVal > 0
            percentText = sprintf('%.1f%%', 100 * yVal);
            if i==2
                aau=0.06
            else
                aau=0.01
            end
            text(x, yVal + aau, percentText, ...
                'HorizontalAlignment', 'center', ...
                'VerticalAlignment', 'bottom', ...
                'FontSize', 14, 'Color', barColor, 'FontWeight', 'bold');
        end
    end
end

set(gcf,'WindowState','maximized')


subplot(1,3,3)
data = Aperc(10:12,indx)';
b = bar(data);  % grouped bars
b(1).FaceColor = [0 0 1];  % Blue
b(2).FaceColor = [1 0 0];  % Red
b(3).FaceColor = [0 1 0];  % Green
title('GPR')
xticklabels(labels(indx))
xtickangle(0)
%legend('$\nu_j$','$\nu_j^{GCMR}$','$\nu_j^{GKnock}$','Interpreter','Latex','Location','northwest')
set(gca,'FontSize',24)

% Add values on top of bars using corresponding bar color
[nGroups, nBars] = size(data);
groupWidth = min(0.8, nBars/(nBars + 1.5));
for i = 1:nBars
    barColor = b(i).FaceColor;
    for j = [1:ndisplayed]
        x = j - groupWidth/2 + (2*i-1) * groupWidth / (2*nBars);
        yVal = data(j,i);
        if yVal > 0
            percentText = sprintf('%.1f%%', 100 * yVal);
            if i==2
                aau=0.06
            else
                aau=0.01
            end
            text(x, yVal + aau,percentText, ...
                'HorizontalAlignment', 'center', ...
                'VerticalAlignment', 'bottom', ...
                'FontSize', 14, 'Color', barColor, 'FontWeight', 'bold');
        end
    end
end
saveas(gcf,['../FiguresUsedinthePaper/',filename,'_PercErr'],'fig')
saveas(gcf,['../FiguresUsedinthePaper/',filename,'_PercErr'],'eps')
saveas(gcf,['../FiguresUsedinthePaper/',filename,'_PercErr'],'jpg')
saveas(gcf,['../FiguresUsedinthePaper/',filename,'_PercErr'],'png')

%%

figure

subplot(3,1,1)
b = bar(AANN');  b(1).FaceColor = [0 0 0]; b(2).FaceColor = [0 0 1]; b(3).FaceColor = [1 0 0]; b(4).FaceColor=[0 1 0];
title('Artificial Neural Network','FontSize',30,'Interpreter','latex')

%legend('$\nu_j$', '$\nu_j^{GCMR}$', '$\nu_j^{GKnock}$','FontSize',28,'Interpreter','latex')
ticks=[1:20];
%xticks(ticks)
%xticklabels(labels(sort(bb(1:7))))
%%xtickangle(45)
%set(gca,'TickLabelInterpreter','none')
%xtickangle(90); % Set orientation to vertical
set(gca,'FontSize',22)
set(gcf,'WindowState','maximized')

subplot(3,1,2)
b = bar(AASVM');  b(1).FaceColor = [0 0 0]; b(2).FaceColor = [0 0 1]; b(3).FaceColor = [1 0 0]; b(4).FaceColor=[0 1 0];
title('SVM','FontSize',30,'Interpreter','latex')
legend('Ground Truth','$\nu_j$', '$\nu_j^{GCMR}$', '$\nu_j^{GKnock}$','FontSize',28,'Interpreter','latex','Location','Northwest')
%xticklabels(labels)
%set(gca,'TickLabelInterpreter','none')
%xtickangle(90); % Set orientation to vertical
set(gca,'FontSize',22)
set(gcf,'WindowState','maximized')

subplot(3,1,3)
b = bar(AAGPR');  b(1).FaceColor = [0 0 0]; b(2).FaceColor = [0 0 1]; b(3).FaceColor = [1 0 0]; b(4).FaceColor=[0 1 0];
title('Gaussian Process Regression','FontSize',30,'Interpreter','latex')
%legend('$\nu_j$', '$\nu_j^{GCMR}$', '$\nu_j^{GKnock}$','FontSize',28,'Interpreter','latex')
%xticklabels(labels)
%set(gca,'TickLabelInterpreter','none')
%xtickangle(90); % Set orientation to vertical
set(gca,'FontSize',22)
set(gcf,'WindowState','maximized')

%
saveas(gcf,['../FiguresUsedinthePaper/',filename,'_All_Breiman'],'fig')
saveas(gcf,['../FiguresUsedinthePaper/',filename,'_All_Breiman'],'eps')
saveas(gcf,['../FiguresUsedinthePaper/',filename,'_All_Breiman'],'jpg')
saveas(gcf,['../FiguresUsedinthePaper/',filename,'_All_Breiman'],'png')

