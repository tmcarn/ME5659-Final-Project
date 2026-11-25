% Balancing bot with DC motor dynamics: input u = motor VOLTAGE [V]
% (same voltage applied to both motors), instead of wheel torque.
%
% Motor model (per motor, inductance neglected - see note at bottom):
%   i     = (V - Kb*N*dpsi) / R           armature current
%   tau_w = N*Kt*i - b_w*dpsi             torque on the wheel (after gearbox)
% where psi = phi - theta is the wheel angle RELATIVE to the body (the motor
% housing is bolted to the body, so back-EMF depends on dpsi, not dphi).
%
% Both motors together (total torque, matching the "both wheels" lumped model):
%   tau = 2*N*Kt/R * V  -  (2*N^2*Kt*Kb/R + b) * dpsi
%         \___ gain ___/    \___ back-EMF + viscous damping ___/

% Variables
syms phi theta dphi dtheta u real

% Params
syms M m r l Ib Iw Ia g positive
syms Ng Kt Kb Rm b positive   % gear ratio, torque const, back-EMF const, resistance, viscous

q = [phi; theta];
dq = [dphi; dtheta];

% Kinematics
p_b = [r*phi + l*sin(theta); l*cos(theta)];
dp_b = jacobian(p_b, q) * dq;

% Kinetic Energy
K_wheel = 0.5*m*(r*dphi)^2 + 0.5*Iw*dphi^2;
% Motor rotor (armature / reflected inertia): spins RELATIVE to the body,
% i.e. with the joint angle psi = phi - theta (this is how MuJoCo models it)
% (Ia = 2 * Ng^2 * J_rotor)
K_motor = 0.5*Ia*(dphi - dtheta)^2;
K_body = 0.5*M*dp_b.'*dp_b + 0.5*Ib*dtheta^2;

K = K_wheel + K_motor + K_body;

% Potential Energy
V = M*g*l*cos(theta);

% Inputs: voltage -> total motor torque (both motors)
dpsi = dphi - dtheta;
tau = 2*Ng*Kt/Rm * u - (2*Ng^2*Kt*Kb/Rm + b) * dpsi;

% Mass Inertial Matrix
D = jacobian(jacobian(K,dq).',dq);
D = simplify(D);

% Coriolis matrix
N = max(size(q));
C = sym(zeros(N));
for k = 1:N
    for j = 1:N
        for i = 1:N
            C(k,j) = C(k,j) + 1/2*(diff(D(k,j),q(i)) + ...
                diff(D(k,i),q(j)) - ...
                diff(D(i,j),q(k)))*dq(i);
        end
    end
end
C = simplify(C);

% Gravity matrix
G = jacobian(V,q).';
G = simplify(G);

% Input matrix: motor torque acts +tau on the wheel, -tau (reaction) on the body
B = [1; -1];

% Linearize for State-Space Model
x_eq = [0; 0; 0; 0];

% Solve for ddq using Inverse Dynamics
ddq = D \ (B*tau - C*dq - G);
ddq = simplify(ddq);

x = [q; dq];
dx = [dq; ddq];

% Jacobian linearization
A_sym = jacobian(dx, x);
B_sym = jacobian(dx, u);

% Evaluate at equilibrium, (zero velocity, zero input)
A0 = simplify(subs(A_sym, [x; u], [x_eq; 0]));
B0 = simplify(subs(B_sym, [x; u], [x_eq; 0]));

% --- Generate numeric functions of the parameters ---
% (different names from the torque model so they don't overwrite balbot_A/B)
params = [M m r l Ib Iw Ia g Ng Kt Kb Rm b];

matlabFunction(A0, 'File', 'balbot_dc_A', 'Vars', {params});
matlabFunction(B0, 'File', 'balbot_dc_B', 'Vars', {params});

% Full nonlinear dynamics for simulation: dx = f(x, V, params)
matlabFunction(dx, 'File', 'balbot_dc_dynamics', 'Vars', {x, u, params});


% ================= Parameter values =================
% Mechanical (from get_params.py / params.json)
M  = 1.17694;      % kg      body mass
m  = 0.265808;     % kg      both wheels
r  = 0.0375;       % m
l  = 0.0747452;    % m
Ib = 0.00378203;   % kg m^2
Iw = 0.000186896;  % kg m^2  both wheels, WITHOUT armature
Ia = 0.002;        % kg m^2  total armature (2 x 0.001 in robot.xml) = 2*Ng^2*J_rotor
g  = 9.81;         % m/s^2

% Motor - PLACEHOLDERS, replace with datasheet / measured values (per motor,
% motor-side constants, i.e. before the gearbox)
Ng   = 30;     % gear ratio
Kt   = 0.01;   % N m / A     torque constant
Kb   = 0.01;   % V s / rad   back-EMF constant (= Kt in SI units)
Rm   = 2.0;    % Ohm         armature resistance
b    = 0.001;  % N m s/rad   total viscous friction at the wheels (both)
Vmax = 12.6;   % V           battery / driver limit

p = [M m r l Ib Iw Ia g Ng Kt Kb Rm b];   % same order as `params` above

fprintf('Stall torque per wheel: %.2f N m,  no-load speed: %.1f rad/s\n', ...
        Ng*Kt*Vmax/Rm, Vmax/(Ng*Kb));
fprintf('Back-EMF damping (total): %.4f N m s/rad\n', 2*Ng^2*Kt*Kb/Rm);

A = balbot_dc_A(p);
B = balbot_dc_B(p);

% State-space model
C = [1 0 0 0; 0 0 0 1]; % output = [phi; dtheta]  (encoder + gyro)
D = zeros(2,1);
sys = ss(A, B, C, D);

eig(A)          % should show one unstable pole (inverted pendulum)
rank(ctrb(A,B)) % should be 4 if controllable
rank(obsv(A,C)) % should be 4 if observable

poles = [-5.1; -5.2; -25.1; -25.2]; % desired closed-loop poles
K = place(A, B, poles)              % state feedback gain, u = -K x is VOLTAGE


% ===================== Simulation =====================
x0 = [0; deg2rad(5); 0; 0];             % initial condition: 5 deg lean
Acl = A - B*K;
ctrl = @(x) max(min(-K*x, Vmax), -Vmax);   % voltage-saturated state feedback

t = linspace(0, 6, 2000);

% Linear (unsaturated)
[y_cl, t] = initial(ss(Acl, B, eye(4), 0), x0, t);
u_cl = -(K*y_cl.').';

% Nonlinear (with voltage saturation)
[t_nl, x_nl] = ode45(@(t,x) balbot_dc_dynamics(x, ctrl(x), p), t, x0);
u_nl = zeros(size(t_nl));
for k = 1:numel(t_nl), u_nl(k) = ctrl(x_nl(k,:).'); end

figure('Name','Closed loop (voltage input)');
subplot(3,1,1)
plot(t, rad2deg(y_cl(:,2)), t_nl, rad2deg(x_nl(:,2)), '--'); grid on
ylabel('\theta [deg]'); legend('linear','nonlinear'); title('Closed loop, V = -Kx')
subplot(3,1,2)
plot(t, y_cl(:,1)*r, t_nl, x_nl(:,1)*r, '--'); grid on
ylabel('x = r\phi [m]')
subplot(3,1,3)
plot(t, u_cl, t_nl, u_nl, '--'); grid on
ylabel('V [V]'); xlabel('t [s]')

fprintf('Peak voltage (nonlinear): %.2f V\n', max(abs(u_nl)));

% ----------------------------------------------------------------------
% Note on inductance: the electrical time constant L/R of a small DC motor
% is typically ~1 ms, much faster than the 5-25 rad/s closed-loop poles, so
% the current is treated as instantaneous. If you want to keep it, add the
% current i as a 5th state:  L*di/dt = V - R*i - Kb*Ng*dpsi,
% with tau = 2*Ng*Kt*i - b*dpsi, and place 5 poles instead of 4.
