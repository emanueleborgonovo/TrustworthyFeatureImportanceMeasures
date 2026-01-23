clc
load('Model99_replicates.mat')
filename = matlab.desktop.editor.getActiveFilename;
[~, filename, ~] = fileparts(filename);
d=99
labels = arrayfun(@(k) sprintf('X_{%d}', k), 1:d, 'UniformOutput', false);
%%

[aa,bb]=sort(AANN(2,:),'Descend');

Aperc=[percerr_LR_nu;
percerr_LR_nuGCMR;
percerr_LR_nuGKnock;
percerr_NN_nu;
percerr_NN_nuGCMR;
percerr_NN_nuGKnock;
percerr_RF_nu;
percerr_RF_nuGCMR;
percerr_RF_nuGKnock];

%%
figure

% --- Linear Model (not stacked) ---
subplot(1,3,1)
data = Aperc(4:6,bb(1:7))';
b = bar(data);  % grouped bars
b(1).FaceColor = [0 0 1];  % Blue
b(2).FaceColor = [1 0 0];  % Red
b(3).FaceColor = [0 1 0];  % Green
title('Artificial NN')
legend('$\nu_j$','$\nu_j^{GCMR}$','$\nu_j^{GKnock}$','Interpreter','Latex','Location','northwest')
xticklabels(labels(bb(1:7)))
set(gca,'FontSize',24)

% Add values on top of bars using corresponding bar color
[nGroups, nBars] = size(data);
groupWidth = min(0.8, nBars/(nBars + 1.5));
for i = 1:nBars
    barColor = b(i).FaceColor;
    for j = 1:nGroups
        x = j - groupWidth/2 + (2*i-1) * groupWidth / (2*nBars);
        yVal = data(j,i);
        if yVal > 0
            percentText = sprintf('%.1f%%', 100 * yVal);
            text(x, yVal + 0.01, percentText, ...
                'HorizontalAlignment', 'center', ...
                'VerticalAlignment', 'bottom', ...
                'FontSize', 14, 'Color', barColor, 'FontWeight', 'bold');
        end
    end
end
% --- Artificial NN (not stacked) ---
subplot(1,3,2)
data = Aperc(1:3,bb(1:7))';
b = bar(data);  % grouped bars
b(1).FaceColor = [0 0 1];  % Blue
b(2).FaceColor = [1 0 0];  % Red
b(3).FaceColor = [0 1 0];  % Green
title('Linear Model')
legend('$\nu_j$','$\nu_j^{GCMR}$','$\nu_j^{GKnock}$','Interpreter','Latex','Location','northwest')
xticklabels(labels(bb(1:7)))
set(gca,'FontSize',18)

% Add values on top of bars using corresponding bar color
[nGroups, nBars] = size(data);
groupWidth = min(0.8, nBars/(nBars + 1.5));
for i = 1:nBars
    barColor = b(i).FaceColor;
    for j = 1:nGroups
        x = j - groupWidth/2 + (2*i-1) * groupWidth / (2*nBars);
        yVal = data(j,i);
        if yVal > 0
            percentText = sprintf('%.1f%%', 100 * yVal);
            text(x, yVal + 0.01, percentText, ...
                'HorizontalAlignment', 'center', ...
                'VerticalAlignment', 'bottom', ...
                'FontSize', 14, 'Color', barColor, 'FontWeight', 'bold');
        end
    end
end

set(gcf,'WindowState','maximized')

subplot(1,3,3)
data = Aperc(7:9,bb(1:7))';
b = bar(data);  % grouped bars
b(1).FaceColor = [0 0 1];  % Blue
b(2).FaceColor = [1 0 0];  % Red
b(3).FaceColor = [0 1 0];  % Green
title('Random Forest')
legend('$\nu_j$','$\nu_j^{GCMR}$','$\nu_j^{GKnock}$','Interpreter','Latex','Location','northwest')
xticklabels(labels(bb(1:7)))
set(gca,'FontSize'18)

% Add values on top of bars using corresponding bar color
[nGroups, nBars] = size(data);
groupWidth = min(0.8, nBars/(nBars + 1.5));
for i = 1:nBars
    barColor = b(i).FaceColor;
    for j = 1:nGroups
        x = j - groupWidth/2 + (2*i-1) * groupWidth / (2*nBars);
        yVal = data(j,i);
        if yVal > 0
            percentText = sprintf('%.1f%%', 100 * yVal);
            text(x, yVal + 0.01, percentText, ...
                'HorizontalAlignment', 'center', ...
                'VerticalAlignment', 'bottom', ...
                'FontSize', 14, 'Color', barColor, 'FontWeight', 'bold');
        end
    end
end



%%

figure

subplot(3,1,1)
b = bar(AANN');  b(1).FaceColor = [0 0 0]; b(2).FaceColor = [0 0 1]; b(3).FaceColor = [1 0 0]; b(4).FaceColor=[0 1 0];
title('Artificial Neural Network','FontSize',30,'Interpreter','latex')

%legend('$\nu_j$', '$\nu_j^{GCMR}$', '$\nu_j^{GKnock}$','FontSize',28,'Interpreter','latex')
ticks=[sort(bb(1:7))];
%xticks(ticks)
%xticklabels(labels(sort(bb(1:7))))
%%xtickangle(45)
%set(gca,'TickLabelInterpreter','none')
%xtickangle(90); % Set orientation to vertical
set(gca,'FontSize',22)
set(gcf,'WindowState','maximized')

subplot(3,1,2)
b = bar(AALR');  b(1).FaceColor = [0 0 0]; b(2).FaceColor = [0 0 1]; b(3).FaceColor = [1 0 0]; b(4).FaceColor=[0 1 0];
title('Linear Model','FontSize',30,'Interpreter','latex')
legend('Ground Truth','$\nu_j$', '$\nu_j^{GCMR}$', '$\nu_j^{GKnock}$','FontSize',28,'Interpreter','latex','Location','Northwest')
%xticklabels(labels)
%set(gca,'TickLabelInterpreter','none')
%xtickangle(90); % Set orientation to vertical
set(gca,'FontSize',22)
set(gcf,'WindowState','maximized')

subplot(3,1,3)
b = bar(AARF');  b(1).FaceColor = [0 0 0]; b(2).FaceColor = [0 0 1]; b(3).FaceColor = [1 0 0]; b(4).FaceColor=[0 1 0];
title('Random Forest','FontSize',30,'Interpreter','latex')
%legend('$\nu_j$', '$\nu_j^{GCMR}$', '$\nu_j^{GKnock}$','FontSize',28,'Interpreter','latex')
%xticklabels(labels)
%set(gca,'TickLabelInterpreter','none')
%xtickangle(90); % Set orientation to vertical
set(gca,'FontSize',22)
set(gcf,'WindowState','maximized')

%
%saveas(gcf,['../FiguresUsedinthePaper/',filename,'_All_Breiman'],'fig')
%saveas(gcf,['../FiguresUsedinthePaper/',filename,'_All_Breiman'],'eps')
%saveas(gcf,['../FiguresUsedinthePaper/',filename,'_All_Breiman'],'jpg')
%saveas(gcf,['../FiguresUsedinthePaper/',filename,'_All_Breiman'],'png')

