% Drost Test Example

clear all; close all; clc;

[chair1_tri,chair1_pts] = ply_read('chair1.ply','tri');
chair1_pts = chair1_pts.';
chair1_tri = chair1_tri.';
% trisurf(chair1_tri,chair1_pts(:,1),chair1_pts(:,2),chair1_pts(:,3), ...
%         'EdgeColor', 'none', 'FaceAlpha', 0.8);
% colormap(gray);
% axis equal;

TR = triangulation(chair1_tri, chair1_pts);
vn = vertexNormal(TR);

% trisurf(TR,'FaceAlpha', 0.8, 'EdgeColor', 'none');
axis equal
hold on
N = 1000;
quiver3(chair1_pts(1:N:end,1),chair1_pts(1:N:end,2),chair1_pts(1:N:end,3),...
     vn(1:N:end,1),vn(1:N:end,2),vn(1:N:end,3),0.5,'color','b');
% quiver3(Xfb(:,1),Xfb(:,2),Xfb(:,3),...
%      vn(:,1),vn(:,2),vn(:,3),0.5,'color','b');
hold off