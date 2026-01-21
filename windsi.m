function [Si,STi,SiV,TiV,Vy,xy]=windsi(Sigma,n,model,trafo,randomsource,gfx)
% WINDSI First and total effects using winding stairs with correlation.
%      [SI,STI]=WINDSI(SIGMA,N,MODEL,TRAFO) computes these sensitivity
%       measures for MODEL taking input parameters with 
%       distribution TRAFO(Normal(0,SIGMA)) for N base samples

% Reference: I.M. Sobol', S. Tarantola, D. Gatelli, S.S. Kucherenko,
%            W. Mauntz: Estimating the approximation error when fixing 
%            unessential factors in global sensitivity analysis,
%            Reliab Eng Syst Safety 92 (2007), 957-960
%            A. Saltelli, P. Annoni, I. Azzini, F. Campolongo, M. Ratto, 
%            S. Tarantola: Variance based sensitivity analysis of model 
%			 output. Design and estimator for the total sensitivity index,
%            Comput. Phys. Comm. 181 (2010), 259-270
%            T. Goda, A simple algorithm for global sensitivity 
%                  analysis with Shapley effects, RESS 2021
%            E. Plischke, G. Rabitti, E. Borgonovo (2021)
%            E. Plischke: An effective algorithm (2010)
%            E. Borgonovo, E. Plischke, C. Prieur: Total Effects 
%            with Constrained Features, Stat. & Comput. (2024, forthcoming)

% written by elmar.plischke@tu-clausthal.de
if(isscalar(Sigma))
    k=Sigma;
    Sigma=eye(k,k);
else
    k=size(Sigma,1);
end
% replace stat toolbox by error function calls
norminv=@(alfa)-sqrt(2)*erfinv(1-2*alfa); 
normcdf=@(x)erfc(-x/sqrt(2))/2;

% create 2 sets of samples
if(nargin>=5 && ~isempty(randomsource))
    if ~isnumeric(randomsource)
        rsource=randomsource(n,2*k);
        ua=rsource(:,1:k);
        zi=norminv(rsource(:,k+(1:k))); % innovation
    else
        [nn,kk]=size(randomsource);
        if(n~=nn) || (k~=kk/2)
            error('IHSSI: incompatible size of randomsource');
        end
        ua=randomsource(:,1:k);
        zi=norminv(randomsource(:,k+(1:k))); % innovation
    end
else
%% use standard uniform random generator
    ua=rand(n,k);
    zi=randn(n,k);  % innovation
%%
end

%% precompute the correlation weights
D=zeros(k,k);
for i=1:k
    ii=[i+1:k,1:i]; 
    C=chol(Sigma(ii,ii));
    D(ii,i)=[C(1:end-1,1:end-1)\C(1:end-1,end); C(end,end)];
end
% C is now Cholesky decomp for transformation

% correlated sample
za=norminv(ua)*C;
za0=za; % testing testing
% use the provided transformation or identity mapping
if(nargin==3)|| isempty(trafo)
% or default to normal?
    xa=normcdf(za); % rank-correlated in [0,1]^k
else
    xa=trafo(normcdf(za));
end

%% model evaluation for reference sample runs
ya=model(xa); evals=n;
if(nargout>=5)
    xy=[xa,ya];
end
%% estimating mean and var
%Ey=mean(ya);
Vy=var(ya);

Si=ones(1,k);STi=ones(1,k);
SSi=zeros(k+1,k);
% do we need to run twice through a winding stairs design to get the first
% order effects?
% or just call easi with the all the sample blocks

% Compute all first order effects from correlated sample
SSi(k+1,:)=easi(xa,ya,8); % cosi needs signal toolbox
%%
for i = 1:k
%% insert innovation at ith parameter
   yb=ya;
   za(:,i)=zi(:,i); % norminv(rand(n,1));
   za(:,i)=za*D(:,i);
   %% testing testing
   %corrcoef([za0,za]) % check if knockoff exchangeability is satisfied
   %%
   xa=trafo(normcdf(za));
   % evaluate model
   ya=model(xa); evals=evals+n;
   SSi(i,:)=easi(xa,ya,8); % all first order effects
   % save output if requested
   if(nargout>=5)
       xy=[xy;xa,ya];
   end
	% difference
    dy=ya-yb;
% for totals, use Jansen's formula
    STi(i)=(dy'*dy)/(2*n*Vy);
% for the error estimate, check with Goda
    Ti2(i)=mean(dy.^4)/(2*Vy).^2;
%%
end
Si=mean(SSi);
SiV=var(SSi);
TiV=(Ti2-STi.^2)/(n-1);
%fprintf('Model evaluations: %d\n',evals);
%Si, Si_
%STi,STi_
return
end

function testwind
%%
% Compare with Kucherenko Tarantola Annoni
model=@(x)sin(x(:,1)).*(1.0+0.1*x(:,3).^4)+7.0*(sin(x(:,2))).^2;
trafo=@(u)(u-.5)*2*pi;
k=3;
n=8192; % 8192*16; %4096; %
rhos=-.9:.05:.9;
Ss=[];Ts=[];
VSs=[];
VTs=[];
SNs=[];TNs=[];
for rho=rhos
    rho
    Sig=eye(k,k);
    Sig(1,3)=rho;Sig(3,1)=rho;    
    %[S,T,VS,VT]=windsi(Sig,n,model,trafo,@sobolpoints); 
    [S,T]=windsi(Sig,n,model,trafo,@mhalton); % mingled Halton sequence
    %[S,T]=windsi(Sig,n,model,trafo,@rand);
    Ss(end+1,:)=S;
    Ts(end+1,:)=T;
    % estimator variances
    % VSs(end+1,:)=VS;
    % VTs(end+1,:)=VT;
    % compare with nearest neighbor
    x=trafo(normcdf(randn(n*(k+1),k)*chol(Sig)));
    y=model(x);
%    [SN,TN,V]=nupsi(x,y);
%    SNs(end+1,:)=SN./V;
% TNs(end+1,:)=TN./V;
    T=tmcnn(x,y); %
    TNs(end+1,:)=T;
end
%%
subplot(1,3,2);
%plot(rhos,Ss,'-','LineWidth',3);hold on
%set(gca,'ColorOrderIndex',1);
plot(rhos,Ts,'-','LineWidth',3);hold off
xlabel('Rank correlation between x_1 and x_3')
ylabel('Sensitivity index');
title('Winding Stairs')
grid on
%try
%    legend({'T_1','T_2','T_3'},'AutoUpdate','off')
%catch
%    legend({'T_1','T_2','T_3'})
%end
subplot(1,3,3);
%plot(rhos,SNs,'-','LineWidth',3);hold on
%set(gca,'ColorOrderIndex',1);
plot(rhos,TNs,'-','LineWidth',3);hold off
xlabel('Rank correlation between x_1 and x_3')
ylabel('Sensitivity index');
title('Nearest Neighbor')
grid on

%% error bars
subplot(1,2,1)
hold on
for i=1:k
patch([rhos';flipud(rhos')],[Ts(:,i)+2*sqrt(VTs(:,i));...
                      flipud(Ts(:,i)-2*sqrt(VTs(:,i)))],[.8,.8,.8],'EdgeColor',[.8,.8,.8])
end
hold off
% Patches behind lines
set(gca,'Children',flipud(get(gca,'Children')));
%%
end
