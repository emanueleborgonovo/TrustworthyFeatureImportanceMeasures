close all
clear all

% Triangular Distributions with Gaussian Copula

n=2000;
u=rand(n,2);
x=iccorrelate([triinv(u(:,1),0,1,3),triinv(u(:,2),0,2,3)],[1,.9;.9,1]);
figure(1)
subplot(3,3,1)
plot(x(:,1),x(:,2),'bo');
set(gca,'FontSize',20)
title('i) Original Data','FontSize',20)
xlabel('X_1','FontSize',20)
ylabel('X_2','FontSize',20)



% Triangular Distributions with Gaussian Copula: Indeed
z=norminv(uniformer(x));
subplot(3,3,2)
plot(z(:,1),z(:,2),'bo')
set(gca,'FontSize',20)
title('ii) Gaussian Transform','FontSize',20)
xlabel('Z_1','FontSize',20)
ylabel('Z_2','FontSize',20)
%axes(h(1))
%hold('on')
%plot([-3,3],[-3,+3]*.9,'r-','LineWidth',3)


% Remove Linear Trend, Permute, Add Trend
subplot(3,3,3)
plot(z(:,1),z(:,2),'o','Color','b');hold on; plot([-3,3],.9*[-3,3],'r-','LineWidth',3)
set(gca,'FontSize',20)
% axes 1
title('iii) Non-parametric E[Z_{2}|Z_{1}]','FontSize',20)
xlabel("Z_1",'FontSize',20)
ylabel("Z_2",'FontSize',20)


%subplot(4,1,2)
%
w=z(:,2)-.9*z(:,1);
subplot(3,3,4)
plot(z(:,1),w,'bo')
m=w(randperm(length(w)));
set(gca,'FontSize',20)
title('iv) Compute Residuals','FontSize',20)
xlabel('Z_{1}','FontSize',20)
ylabel('Z_{2}-E[Z_{2}|Z_{1}]','FontSize',20)

%subplot(4,1,3)
subplot(3,3,5)
plot(z(:,1),m,'gs')
set(gca,'FontSize',20)
title('v) Permute Residuals','FontSize',20);
xlabel('Z_{1}','FontSize',20)
ylabel('Z^{\pi}_{2}-E[Z_{2}|Z_{1}]','FontSize',20)
%subplot(4,1,4)

subplot(3,3,6)
zz=[z(:,1),.9*z(:,1)+m];
plot(zz(:,1),zz(:,2),'gs');hold on; plot([-3,3],.9*[-3,3],'r-','LineWidth',3)
set(gca,'FontSize',20)
title('vi) Add Back E[Z_{j}|Z_{i}]','FontSize',20);
xlabel('Z_1','FontSize',20)
ylabel('Z_2^{\pi}','FontSize',20)


% Still Gaussian
subplot(3,3,7)
plot(zz(:,1),zz(:,2),'gs');
set(gca,'FontSize',20)
title('vii) Permuted Gaussian','FontSize',20);
xlabel('Z_1','FontSize',20)
ylabel('Z_2^{\pi}','FontSize',20)


% Back to Original Distribution
xs=sort(x(:,2));
%scatterhist(x(:,1),xs(ceil(normcdf(zz(:,2))*length(x))),'Color','g') %wo interpolation
%pause
xx=[x(:,1),quantile(x(:,2),normcdf(zz(:,2)))];
figure(1)
subplot(3,3,8)
plot(x(:,1),xx(:,2),'gs') % with
set(gca,'FontSize',20)
title('viii) Map Back','FontSize',20);
xlabel('X_1','FontSize',20)
ylabel('X_2^{\pi}','FontSize',20)

figure(1)
subplot(3,3,9)
plot(x(:,1),x(:,2),'bo',xx(:,1),xx(:,2),'gs')
set(gca,'FontSize',20)
title('ix) Permuted and Original','FontSize',20);
xlabel('X_1','FontSize',20)
ylabel('X_2,X_2^{\pi}','FontSize',20)

%%
%Comparison to free permutations
% Just scatter
figure
subplot(1,2,1)
plot(x(:,1),x(:,2),'bo')
xlim([0.00 3.00])
ylim([0.00 3.00])
set(gca,'FontSize',16)
title('$\mathbf{X}$','FontSize',20,'interpreter','latex')
xlabel('$X_1$','FontSize',18,'interpreter','latex')
ylabel('$X_2$','FontSize',18,'interpreter','latex')
subplot(1,2,2)
xxx=x;
xxx(:,1)=x(randperm(length(x(:,1))),1);
hold on
plot(x(:,1),x(:,2),'bo',xxx(:,1),xxx(:,2),'r*')
xlim([0.00 3.00])
ylim([0.00 3.00])
title('$\mathbf{X}$ and $\mathbf{X}^\prime$-unrestricted','FontSize',20,'interpreter','latex');
%legend('Original ($\mathbf{X}$)','New ($\mathbf{X}^\prime)$','interpreter','latex','Fontsize',18)
set(gca,'FontSize',16)
xlabel('$X_1$','FontSize',18,'interpreter','latex')
ylabel('$X_2$','FontSize',18,'interpreter','latex')
%Conditional Model Relyiance without gaussian transformation
%% Remove Linear Trend, Permute, Add Trend
hold off
figure
% Find residuals without transformation
mdl = fitlm(x(:,1),x(:,2));
%coefficients=mdl.Coefficients.Estimate;
%coefficients
plot(x(:,1),x(:,2),'bo');hold on; plot(x(:,1),predict(mdl,x(:,1)),'r-','LineWidth',3)
% axes 1
title('Non-parametric E[Z_{2}|Z_{1}]','FontSize',20)
xlim([0.00 3.00])
ylim([0.00 3.00])
xlabel("X_1",'FontSize',20)
ylabel("X_2",'FontSize',20)

% Residuals
r2=x(:,2)-predict(mdl,x(:,1));
r2p=r2(randperm(length(x(:,2))));
x2p=r2p+predict(mdl,x(:,1));
xp=[x(:,1) x2p];

figure
plot(x(:,1),x(:,2),'bo');hold on; plot(x(:,1),x2p,'gs','LineWidth',3)
yline(3)
yline(0)

figure
subplot(1,4,1)
plot(x(:,1),x(:,2),'ko')
xlim([0.00 3.00])
ylim([0.00 4.00])
set(gca,'FontSize',24)
title('$\mathbf{X}$','FontSize',20,'interpreter','latex')
legend('Original','FontSize',20,'interpreter','latex')
xlabel('$X_1$','FontSize',18,'interpreter','latex')
ylabel('$X_2$','FontSize',18,'interpreter','latex')
legend('Original','FontSize',14,'interpreter','latex')


subplot(1,4,2)
xxx=x;
xxx(:,1)=x(randperm(length(x(:,1))),1);
hold on
plot(x(:,1),x(:,2),'ko',xxx(:,1),xxx(:,2),'r*')
xlim([0.00 3.00])
ylim([0.00 4.00])
title('$\mathbf{X}$, $\mathbf{X}^\prime$-unr.','FontSize',20,'interpreter','latex');
%legend('Original ($\mathbf{X}$)','New ($\mathbf{X}^\prime)$','interpreter','latex','Fontsize',18)
set(gca,'FontSize',24)
xlabel('$X_1$','FontSize',18,'interpreter','latex')
ylabel('$X_2$','FontSize',18,'interpreter','latex')
legend('Original','Permuted','FontSize',14,'interpreter','latex')

subplot(1,4,3)
plot(x(:,1),x(:,2),'ko');hold on; plot(x(:,1),x2p,'bs','LineWidth',3)
yline(3,'r-.','LineWidth',3)
yline(0)
title('$\mathbf{X}$, $\mathbf{X}^\prime$-CMR','FontSize',20,'interpreter','latex');
%legend('Original ($\mathbf{X}$)','New ($\mathbf{X}^\prime)$','interpreter','latex','Fontsize',18)
set(gca,'FontSize',24)
xlabel('$X_1$','FontSize',18,'interpreter','latex')
ylabel('$X_2$','FontSize',18,'interpreter','latex')
legend('Original','CMR','FontSize',14,'interpreter','latex')

subplot(1,4,4)
plot(x(:,1),x(:,2),'ko',xx(:,1),xx(:,2),'gs')
title('$\mathbf{X}$, $\mathbf{X}^\prime$-GCMR','FontSize',20,'interpreter','latex');
xlim([0.00 3.00])
ylim([0.00 4.00])
set(gca,'FontSize',24)
legend('Original','G-CMR','FontSize',20,'interpreter','latex')
xlabel('$X_1$','FontSize',18,'interpreter','latex')
ylabel('$X_2$','FontSize',18,'interpreter','latex')
legend('Original','Permuted','FontSize',16,'interpreter','latex')

