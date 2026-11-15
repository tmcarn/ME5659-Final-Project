% Variables
syms phi theta dphi dtheta u real

% Params
syms M m r l Ib Iw g positive

q = [phi; theta];
dq = [dphi; dtheta];
ddq = [ddphi; ddtheta];

% Kinematics
p_b = [r*phi + l*sin(theta); l*cos(theta)];
dp_b = jacobian(p_b, q) * dq;

% Kinetic Energy
K_wheel = 0.5*m*(r*dphi)^2 + 0.5*Iw*dphi^2;
K_body = 0.5*M*dp_b.'*dp_b + 0.5*Ib*dtheta^2;

K = K_wheel + K_body;

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

A0 = simplify(A0)
B0 = simplify(B0)



