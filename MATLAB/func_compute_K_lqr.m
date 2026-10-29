
function K_lqr = func_compute_K_lqr(params)

set_path

[A0, B0] = func_compute_A0_B0([0,0],[0,0],params);

% Tuning weights
Q = diag([20, 20, 20, 20]);   % penalize [x, theta, dx, dtheta]
R = 10;                       % penalize input force

% Compute gain
K_lqr = lqr(A0, B0, Q, R)

end