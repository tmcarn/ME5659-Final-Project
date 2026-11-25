% Variables
syms phi theta dphi dtheta u real

% Params
syms M m r l Ib Iw Ia g positive

q = [phi; theta];
dq = [dphi; dtheta];

% Kinematics
p_b = [r*phi + l*sin(theta); l*cos(theta)];
dp_b = jacobian(p_b, q) * dq;

% Kinetic Energy
K_wheel = 0.5*m*(r*dphi)^2 + 0.5*Iw*dphi^2;
% Motor rotor (armature / reflected inertia): spins RELATIVE to the body,
% i.e. with the joint angle psi = phi - theta (this is how MuJoCo models it)
K_motor = 0.5*Ia*(dphi - dtheta)^2;
K_body = 0.5*M*dp_b.'*dp_b + 0.5*Ib*dtheta^2;

K = K_wheel + K_motor + K_body;

% Potential Energy
V = M*g*l*cos(theta);

% Inputs
tau = u;

% Mass Inertial Matrix
D = jacobian(jacobian(K,dq).',dq); 
D = simplify(D)

% Coriolis matrix
N = max(size(q));
syms C
for k = 1:N
    for j = 1:N
        C(k,j) = 0*g;
        for i = 1:N
            C(k,j) = C(k,j) + 1/2*(diff(D(k,j),q(i)) + ...
                diff(D(k,i),q(j)) - ...
                diff(D(i,j),q(k)))*dq(i);
        end
    end
end
C = simplify(C)

% Gravity matrix
G = jacobian(V,q).'; 
G = simplify(G)

% Input matrix
B = [1; -1];


% Linerarize for State-Space Model
x_eq = [0; 0; 0; 0];

% Solve for ddq using Inverse Dynamics
ddq = D \ (B*u - C*dq - G);
ddq = simplify(ddq);

x = [q; dq];
dx = [dq; ddq];

% Jacobian linearization
A_sym = jacobian(dx, x);
B_sym = jacobian(dx, u);

% Evaluate at equilibrium, (zero velocity, zero input)
A0 = subs(A_sym, [x; u], [x_eq; 0]);
B0 = subs(B_sym, [x; u], [x_eq; 0]);

A0 = simplify(A0);
B0 = simplify(B0);

% --- Generate numeric functions of the parameters ---
params = [M m r l Ib Iw Ia g];

matlabFunction(A0, 'File', 'balbot_A', 'Vars', {params});
matlabFunction(B0, 'File', 'balbot_B', 'Vars', {params});

% Full nonlinear dynamics for simulation: dx = f(x, u, params)
matlabFunction(dx, 'File', 'balbot_dynamics', 'Vars', {x, u, params});


% Parameter values (from get_params.py / params.json)
  M  = 1.17694  % kg      body mass
  m  = 0.265808  % kg     both wheels
  r  = 0.0375  % m
  l  = 0.0747452  % m
  Ib = 0.00378203  % kg m^2
  Iw = 0.000186896  % kg m^2   both wheels, WITHOUT armature
  Ia = 0.002  % kg m^2         total armature (2 x 0.001 in robot.xml)
  g  = 9.81  % m/s^2

p = [M m r l Ib Iw Ia g];   % same order as `params` above

A = balbot_A(p);
B = balbot_B(p);

% State-space model, output = [phi; theta]
% C = [eye(2) zeros(2)];
C = [1 0 0 0; 0 0 0 1]; % output = [phi; dtheta]
D = zeros(2,1);
sys = ss(A, B, C, D);

eig(A)          % should show one unstable pole (inverted pendulum)
rank(ctrb(A,B)) % should be 4 if controllable
rank(obsv(A,C)) % should be 4 if observable

poles = [-5.1; -5.2; -25.1; -25.2]; % desired closed-loop poles
K = place(A, B, poles) % state feedback gain


% ===================== Simulation =====================
x0 = [0; deg2rad(5); 0; 0];      % initial condition: 5 deg lean
Acl = A - B*K;
ctrl = @(x) -K*x;                % state feedback law

% ---------- 2. Closed loop ----------
t = linspace(0, 6, 2000);

% Linear
[y_cl, t] = initial(ss(Acl, B, eye(4), 0), x0, t);
u_cl = -(K*y_cl.').';

% Nonlinear
[t_nl, x_nl] = ode45(@(t,x) balbot_dynamics(x, ctrl(x), p), t, x0);
u_nl = -(K*x_nl.').';

figure('Name','Closed loop');
subplot(3,1,1)
plot(t, rad2deg(y_cl(:,2)), t_nl, rad2deg(x_nl(:,2)), '--'); grid on
ylabel('\theta [deg]'); legend('linear','nonlinear'); title('Closed loop, u = -Kx')
subplot(3,1,2)
plot(t, y_cl(:,1)*r, t_nl, x_nl(:,1)*r, '--'); grid on
ylabel('x = r\phi [m]')
subplot(3,1,3)
plot(t, u_cl, t_nl, u_nl, '--'); grid on
ylabel('u [N·m]'); xlabel('t [s]')

fprintf('Peak torque (nonlinear): %.3f N·m\n', max(abs(u_nl)));