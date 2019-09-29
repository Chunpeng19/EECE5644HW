% EECE5644 HW1 Question 2
clear all, close all

% parameters
a1 = 0;
b1 = 1;
a2 = 1;
b2 = 2;

% log-likelihood-ratio function
syms x;
l = abs(x - a2)/b2 - abs(x - a1)/b1 + log(b2 / b1);

% function plot
fplot(l,'LineWidth',1.5)
xlabel('$x$','Interpreter','latex')
ylabel('$\ell (x)$','Interpreter','latex')
title('Plot of log-likelihood-ratio function')