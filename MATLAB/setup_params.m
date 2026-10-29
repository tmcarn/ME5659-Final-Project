set_path

m = 1.18; % kg
M = 0.25; % kg
l = 0.075; % m
g = 9.81; % m/s^2

params = [m, M, l, g];

%% LQR Parameters
[A0, B0] = func_compute_A0_B0([0,0],[0,0],params);

% Tuning weights
Q = diag([20, 20, 20, 20]);   % penalize [x, theta, dx, dtheta]
R = 10;                       % penalize input force

% Controller Gains
K_lqr = lqr(A0, B0, Q, R)

%% DC Motor Params
r = 0.065; % wheel radius (m)
n = 21.5; % gear reduction ratio

R = 2.0; % Ohms
L = 0.5; % Henrys
Km = 0.1; % torque constant
Kb = 0.1; % back emf constant
Kf = 0.2; % Nms
J = 0.02; % kg .mˆ2/ s ˆ2

Vbat = 12.6; % Battery Voltage Range

% DC Controller Gains
Kp = 43;
Ki = 1250;

% Sensor STD
std = 0.01;

