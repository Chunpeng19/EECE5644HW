% EECE5644 Exam1 Question 3
clear all, close all

% parameters
d = 4; % dimension of paramter
N = 10; % # of samples
M = 100; % # of tests
K = 21; % # of gamma
a = -1; b = 1; % range of x distribution
sigma = 1;

% for Index = 1:4
%     sigma = 10^(2*(Index-3)+1);
%     errorPlot(d,N,M,K,a,b,sigma,Index)
% end

% function errorPlot(d,N,M,K,a,b,sigma,Index)
% error
e = zeros(M,1);
e_max = zeros(K,1);
e_min = zeros(K,1);
e_25 = zeros(K,1);
e_75 = zeros(K,1);
e_med = zeros(K,1);
gamma = zeros(K,1);

for k = 1:K
    
    gamma(k) = 10^(k-(K+1)/2);
    
    % w
    w = zeros(d,1);
    r = a + (b-a).*rand(d-1,1);
    w(1) = mvnrnd(0,gamma(k)^2);
    w(2) = -w(1)*(r(1)+r(2)+r(3));
    w(3) = w(1)*(r(1)*r(2)+r(2)*r(3)+r(3)*r(1));
    w(4) = -w(1)*r(1)*r(2)*r(3);

    for j = 1:M
        % v distribution
        x = a + (b-a).*rand(N,1);
        v = mvnrnd(0,sigma^2,N);

        % y distribution
        y = (w'*[x.^3 x.^2 x ones(N,1)]')'+v;

        % MAP estimation
        temp1 = zeros(d,d);
        for i = 1:N
            temp1 = temp1 + [x(i)^3 x(i)^2 x(i) 1]'*[x(i)^3 x(i)^2 x(i) 1];
        end
        A = gamma(k)^2*temp1 + sigma^2*eye(d);
        temp2 = zeros(d,1);
        for i = 1:N
            temp2 = temp2 + y(i)*[x(i)^3 x(i)^2 x(i) 1]';
        end
        B = temp2*gamma(k)^2;
        w_MAP = A\B;

        e(j) = norm(w_MAP-w)^2;
    end

    e_max(k) = max(e);
    e_min(k) = min(e);
    e_25(k) = prctile(e,25);
    e_75(k) = prctile(e,75);
    e_med(k) = median(e);

end

figure(3)
% subplot(2,2,Index)
loglog(gamma,e_min,'LineWidth',1.5)
grid on
hold on
loglog(gamma,e_25,'LineWidth',1.5)
loglog(gamma,e_med,'LineWidth',1.5)
loglog(gamma,e_75,'LineWidth',1.5)
loglog(gamma,e_max,'LineWidth',1.5)

xlabel('gamma')
ylabel('squared-error')
legend('minimum error','25% error','median error','75% error','maximum error','Location','southeast')
title(['Squared-error values for different gamma with sigma = ' num2str(sigma)])
hold off
% end
