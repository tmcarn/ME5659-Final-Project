function [D,C,G]= func_compute_D_C_G(q,dq,param)
%%%%%%  func_compute_D_C_G.m
%%%%  04/28/26
%%%%
%%%%
%%%%
%Inputs
x=q(1);
theta=q(2);
%%%%
%%%%
dx=dq(1);
dtheta=dq(2);
%%%%
%%%%
m=param(1);
M=param(2);
l=param(3);
g=param(4);
%%%%
%%%%
%%%%
%%%%
D=zeros(2,2);
D(1,1) = M + m;
D(1,2) = l*m*cos(theta);
D(2,1) = l*m*cos(theta);
D(2,2) = l^2*m;
%%%%
%%%%
C=zeros(2,2);
C(1,1) = 0;
C(1,2) = -dtheta*l*m*sin(theta);
C(2,1) = 0;
C(2,2) = 0;
%%%%
%%%%
G=zeros(2,1);
G(1,1) = 0;
G(2,1) = g*l*m*sin(theta);
%%%%
%%%%
%%End of code