close all
% grab examples points
R     = 15;
I     = 45;

m_r   = chair1_pts_down(R,:);
n_r_m = chair1_vn_down(R,:);
m_i   = chair1_pts_down(I,:);

% plot initial scene
figure, plot3([m_i(1) m_r(1) m_r(1)+n_r_m(1)],[m_i(2) m_r(2) m_r(2)+n_r_m(2)],[m_i(3) m_r(3) m_r(3)+n_r_m(3)],'k')
% label reference and comparison point
hold on, plot3(m_r(1),m_r(2),m_r(3),'xr')
hold on, plot3(m_i(1),m_i(2),m_i(3),'og')

% canonical coords
hold on, plot3([0 1],[0 0],[0 0],'r')
hold on, plot3([0 0],[0 1],[0 0],'b')
hold on, plot3([0 0],[0 0],[0 1],'g')

% generate model to global calculation
transm = trans(-1*[m_r 1]);
rot_y = roty(atan2(n_r_m(3), n_r_m(1)));
n_tmp = rot_y * [n_r_m.';1];
rot_z = rotz(-1*atan2(n_tmp(2), n_tmp(1)));
T_m_g = rot_z * rot_y * transm;

% transform coordinates
temp       = [m_i; m_r; m_r+n_r_m]';
temp       = [temp;ones(1,3)];
new_coords = T_m_g*temp;
m_in       = new_coords(1:3,1)';
m_rn       = new_coords(1:3,2)';
n_r_mn     = new_coords(1:3,3)';

%Visualize
hold on, plot3([m_in(1) m_rn(1) m_rn(1)+n_r_mn(1)],[m_in(2) m_rn(2) m_rn(2)+n_r_mn(2)],[m_in(3) m_rn(3) m_rn(3)+n_r_mn(3)],'k')
hold on, plot3(m_in(1),m_in(2),m_in(3),'og')
hold on, plot3(m_rn(1),m_rn(2),m_rn(3),'xr')
cameratoolbar
