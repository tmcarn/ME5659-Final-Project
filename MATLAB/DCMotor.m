%% DC motor model parameters
R = 2.0; % Ohms
L = 0.5; % Henrys
Km = 0.1; % torque constant
Kb = 0.1; % back emf constant
Kf = 0.2; % Nms
J = 0.02; % kg .mˆ2/ s ˆ2

% Transfer Functions
h1 = tf(Km, [L R]) % Armature
h2 = tf(1, [J Kf]) % EOM

dcm = ss(h2)*[h1, 1];
dcm = feedback(dcm, Kb, 1, 1);

stepplot(dcm(1))

