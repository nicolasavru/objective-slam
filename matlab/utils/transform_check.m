close all
rng('shuffle')
% generate some surrogate data

% Mapping model test point into global frame
m_r   = randn(1,3);
n_r_m = randn(1,3);
n_r_m = n_r_m./norm(n_r_m);
m_i   = randn(1,3);

% plot initial scene
figure, plot3([m_i(1) m_r(1) m_r(1)+n_r_m(1)],[m_i(2) m_r(2) m_r(2)+n_r_m(2)],[m_i(3) m_r(3) m_r(3)+n_r_m(3)],'k')
% label reference and comparison point
hold on, plot3(m_r(1),m_r(2),m_r(3),'xr')
plot3(m_r(1)+n_r_m(1),m_r(2)+n_r_m(2),m_r(3)+n_r_m(3),'^r')
plot3(m_i(1),m_i(2),m_i(3),'or')

% canonical coords
plot3([0 1],[0 0],[0 0],'r')
plot3([0 0],[0 1],[0 0],'b')
plot3([0 0],[0 0],[0 1],'g')

% generate model to global calculation
transm = trans(-1*m_r);
rot_y = roty(atan2(n_r_m(3), n_r_m(1)));
n_tmp = rot_y * [n_r_m.';1];
rot_z = rotz(-1*atan2(n_tmp(2), n_tmp(1)));
T_m_g = rot_z * rot_y * transm;

% transform coordinates
temp       = [m_i; m_r; m_r + n_r_m]';
temp       = [temp;ones(1,3)];
new_coords = T_m_g*temp;
m_in       = new_coords(1:3,1)';
m_rn       = new_coords(1:3,2)';
n_r_mn     = new_coords(1:3,3)';

%Visualize
plot3([m_in(1) m_rn(1) m_rn(1)+n_r_mn(1)],[m_in(2) m_rn(2) m_rn(2)+n_r_mn(2)],[m_in(3) m_rn(3) m_rn(3)+n_r_mn(3)],'k')
plot3(m_rn(1),m_rn(2),m_rn(3),'xr')
plot3(m_rn(1)+n_r_mn(1),m_rn(2)+n_r_mn(2),m_rn(3)+n_r_mn(3),'^r')
plot3(m_in(1),m_in(2),m_in(3),'or')

% Mapping scene test point into global scene 
% (assuming it correponds to our model point

% generate a random transform
rot_xt  = rotx(2*pi*rand);
rot_yt  = roty(2*pi*rand);
rot_zt  = rotz(2*pi*rand);
trans_t = trans(randn,randn,randn);
% apply transformation to generate scene test point
temp    = trans_t*rot_xt*rot_yt*rot_zt*[m_in 1; m_rn 1; n_r_mn 1]';

s_i   = temp(1:3,1)';
s_r   = temp(1:3,2)';
n_r_s = temp(1:3,3)'-s_r;

% plot out initial scene point
plot3([s_i(1) s_r(1) s_r(1)+n_r_s(1)],[s_i(2) s_r(2) s_r(2)+n_r_s(2)],[s_i(3) s_r(3) s_r(3)+n_r_s(3)],'m')
plot3(s_r(1),s_r(2),s_r(3),'xc')
plot3(s_r(1)+n_r_s(1),s_r(2)+n_r_s(2),s_r(3)+n_r_s(3),'^c')
plot3(s_i(1),s_i(2),s_i(3),'oc')

% Map scene test point into global frame
% generate model to global calculation
transm = trans(-1*s_r);
rot_y = roty(atan2(n_r_s(3), n_r_s(1)));
n_tmp = rot_y * [n_r_s.';1];
rot_z = rotz(-1*atan2(n_tmp(2), n_tmp(1)));
T_s_g = rot_z * rot_y * transm;

% transform coordinates
temp       = [s_i; s_r; s_r + n_r_s]';
temp       = [temp;ones(1,3)];
new_coords = T_s_g*temp;
s_in       = new_coords(1:3,1)';
s_rn       = new_coords(1:3,2)';
n_r_sn     = new_coords(1:3,3)';

plot3([s_in(1) s_rn(1) s_rn(1)+n_r_sn(1)],[s_in(2) s_rn(2) s_rn(2)+n_r_sn(2)],[s_in(3) s_rn(3) s_rn(3)+n_r_sn(3)],'m')
plot3(s_rn(1),s_rn(2),s_rn(3),'xc')
plot3(s_rn(1)+n_r_sn(1),s_rn(2)+n_r_sn(2),s_rn(3)+n_r_sn(3),'^c')
plot3(s_in(1),s_in(2),s_in(3),'oc')

% calculate alpha and visualize
point1 = T_m_g*[m_i 1].';
point2 = T_s_g*[s_i 1].';

% You have to project the vectors point1 and point2 onto the yz-plane
w     = [1 0 0]';
u     = point1(1:3);
v     = point2(1:3);
u_hat = u-w*w'*u;
v_hat = v-w*w'*v;
alpha = atan2(w'*cross(u_hat,v_hat),u_hat'*v_hat);

% generate rotation matrix for rotating m_i into s_i
m_i_final = rotx(alpha)*[m_in 1]';
m_i_final = m_i_final(1:3)';
plot3(m_i_final(1),m_i_final(2),m_i_final(3),'or','markerfacecolor','r')
axis equal
cameratoolbar