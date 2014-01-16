function [ T_m_g, T_s_g, alpha ] = trans_model_scene(m_r, n_r_m, m_i, s_r, n_r_s, s_i)
%trans_model_scene Transformation between model and scene coordinates
%   Assume that the 2 point pairs have similar features, without that
%   assumption alpha is not guaranteed to be in the y-z plane, and all
%   logic falls apart

% transm = trans(-1*[m_r 1]);
% rot_y = roty(atan2(n_r_m(3), n_r_m(1)));
% rot_z = rotz(-1*atan2(n_r_m(2), n_r_m(1)));
% T_m_g = rot_z * rot_y * transm;

transm = trans(-1*m_r);
rot_y = roty(atan2(n_r_m(3), n_r_m(1)));
n_tmp = rot_y * [n_r_m.';1];
rot_z = rotz(-1*atan2(n_tmp(2), n_tmp(1)));
T_m_g = rot_z * rot_y * transm;

% transm = trans(-1*s_r);
% rot_y = roty(-1*atan2(n_r_s(3), n_r_s(1)));
% rot_z = rotz(-1*atan2(n_r_s(2), n_r_s(1)));
% T_s_g = rot_z * rot_y * transm;

transm = trans(-1*s_r);
rot_y = roty(atan2(n_r_s(3), n_r_s(1)));
n_tmp = rot_y * [n_r_s.';1];
rot_z = rotz(-1*atan2(n_tmp(2), n_tmp(1)));
T_s_g = rot_z * rot_y * transm;

point1 = T_m_g*[m_i 1].';
point2 = T_s_g*[s_i 1].';
alpha = acos(dot(point1,point2) / (norm(point1)*norm(point2)));

end

