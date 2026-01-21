n=2000;
x=[randi([0,2],2000,1),rand(2000,1)];

for rho=.5:.05:.95
 xx=iccorrelate(x,[1,rho;rho,1]);
 corr(xx)
 subplot(1,3,1);
 hist(xx(xx(:,1)==0,2),20);a=axis;a(1)=0;a(2)=1;axis(a);
  subplot(1,3,2);
 hist(xx(xx(:,1)==1,2),20);a=axis;a(1)=0;a(2)=1;axis(a);
  subplot(1,3,3);
 hist(xx(xx(:,1)==2,2),20);a=axis;a(1)=0;a(2)=1;axis(a);
 pause(1);
end
%%
for rho=.5:.05:.95
 xx=iccorrelate(x,[1,rho;rho,1]);
 corr(xx)
 yy=norminv(xx(:,2),0,1);
 subplot(1,3,1);
 hist(yy(xx(:,1)==0),20);a=axis;a(1)=-3;a(2)=3;axis(a);
  subplot(1,3,2);
 hist(yy(xx(:,1)==1),20);a=axis;a(1)=-3;a(2)=3;axis(a);
  subplot(1,3,3);
 hist(yy(xx(:,1)==2),20);a=axis;a(1)=-3;a(2)=3;axis(a);
 pause(1);
end

% DO the mean in each category; then subtract the mean, redistribute and
% add the mean again
%%
i0=xx(:,1)==0;
n0=mean(xx(i0,2));
m0=mean(yy(i0));
z0=yy(i0)-m0;
%s0=n0+normcdf(z0(randperm(length(z0))),0,1)-.5;

i1=xx(:,1)==1;
n1=mean(xx(i1,2));
m1=mean(yy(i1));
z1=yy(i1)-m1;
%s1=n1+normcdf(z1(randperm(length(z1))),0,1)-.5;

i2=xx(:,1)==2;
n2=mean(xx(i2,2));
m2=mean(yy(i2));
z2=yy(i2)-m2;
%s2=n2+normcdf(z2(randperm(length(z2))),0,1)-.5;

zz=[xx(i0,2)-n0;xx(i1,2)-n1;xx(i2,2)-n2];
ii=randperm(length(zz));
ss=zz(ii)+[ones(sum(i0),1)*n0;ones(sum(i1),1)*n1;ones(sum(i2),1)*n2];

zz=[z0;z1;z2];
qq=normcdf(zz(ii)+[ones(sum(i0),1)*m0;ones(sum(i1),1)*m1;ones(sum(i2),1)*m2]);
%%
figure
%plot(xx(:,1),xx(:,2),'ko',xx(:,1)+.05,normcdf(yy),'g+');
%
%plot(xx(:,1),xx(:,2),'b*',sort(xx(:,1))+.05,[s0;s1;s2],'r.')
plot(xx(:,1),xx(:,2),'ob',sort(xx(:,1))+.05,ss,'r*',sort(xx(:,1))-.05,qq,'gs','LineWidth',3,'MarkerSize',7)
yline(0,'-.','LineWidth',2)
yline(1,'-.','LineWidth',2)
set(gca,'XTick',0:2)
xlabel('X_1','FontSize',24)
ylabel('X_2','FontSize',24)
legend('Original Data','Without Gaussian Transformation', 'With Gaussian Transformation','Fontsize',24)
set(gca,'FontSize',24)