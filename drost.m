% Drost Test Example

% clear all; close all; clc;

[chair1_tri,chair1_pts] = ply_read('chair1.ply','tri');
chair1_pts = chair1_pts.';
chair1_tri = chair1_tri.';
% trisurf(chair1_tri,chair1_pts(:,1),chair1_pts(:,2),chair1_pts(:,3), ...
%         'EdgeColor', 'none', 'FaceAlpha', 0.8);
% colormap(gray);
% axis equal;

TR = triangulation(chair1_tri, chair1_pts);
chair1_vn = vertexNormal(TR);

disp('Chair Model Loaded and Normals Computed');

N = 10000;

[scene1_tri,scene1_pts] = ply_read('scene1.ply','tri');
scene1_pts = scene1_pts.';
scene1_tri = scene1_tri.';


TR = triangulation(scene1_tri, scene1_pts);
scene1_vn = vertexNormal(TR);

disp('Scene Model Loaded and Normals Computed');

% trisurf(TR,'FaceAlpha', 0.8, 'EdgeColor', 'none');
% axis equal
% hold on
% 
% quiver3(chair1_pts(1:N:end,1),chair1_pts(1:N:end,2),chair1_pts(1:N:end,3),...
%      vn(1:N:end,1),vn(1:N:end,2),vn(1:N:end,3),0.5,'color','b');
% quiver3(Xfb(:,1),Xfb(:,2),Xfb(:,3),...
%      vn(:,1),vn(:,2),vn(:,3),0.5,'color','b');
% hold off

chair1_pts_down = chair1_pts(1:N:end,:);
chair1_vn_down = chair1_vn(1:N:end,:);
scene1_pts_down = scene1_pts(1:N:end,:);
scene1_vn_down = scene1_vn(1:N:end,:);

disp('Downsampled');

chairMap = model_description(chair1_pts_down, chair1_vn_down);

disp('Computed chair map');

[max_alpha, transform] = voting_scheme(chairMap, chair1_pts_down, chair1_vn_down, scene1_pts_down, ...
                                       scene1_vn_down);

chair_new_pts = transform*[chair1_pts ones(size(chair1_pts,1),1)].';
chair_new_pts = chair_new_pts(1:3,:).';
hold on;
trisurf(chair1_tri,chair1_pts(:,1),chair1_pts(:,2),chair1_pts(:,3), 'EdgeColor', 'green');
trisurf(chair1_tri,chair_new_pts(:,1),chair_new_pts(:,2),chair_new_pts(:,3));
trisurf(scene1_tri,scene1_pts(:,1),scene1_pts(:,2),scene1_pts(:,3), ...
        'EdgeColor', 'none', 'FaceAlpha', 0.8);
axis equal;


                                     