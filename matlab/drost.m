% Drost Test Example

clear all; close all; clc;

[chair1_tri,chair1_pts] = ply_read('../ply/chair1.ply','tri');
%[~,chair1_pts] = ply_read('../ply/chair1_centered.ply','tri');
chair1_pts = chair1_pts.';
chair1_tri = chair1_tri.';
% trisurf(chair1_tri,chair1_pts(:,1),chair1_pts(:,2),chair1_pts(:,3), ...
%         'EdgeColor', 'none', 'FaceAlpha', 0.8);
% colormap(gray);
% axis equal;

TR = triangulation(chair1_tri, chair1_pts);
chair1_vn = vertexNormal(TR);

%[chair1,chair1_comments] = ply_read('../../scan_tmp/cup/object.ply')
%chair1_pts = [chair1.vertex.x, chair1.vertex.y, chair1.vertex.z];
%chair1_vn = [chair1.vertex.nx, chair1.vertex.ny, chair1.vertex.nz];

disp('Chair Model Loaded and Normals Computed');

model_N = 2500;
scene_N = 1000;

[scene1_tri,scene1_pts] = ply_read('../ply/scene1_trimmed2.ply','tri');
%[scene1_tri,scene1_pts] = ply_read('../ply/chair1.ply','tri');
%scene1_pts = [scene1.vertex.x, scene1.vertex.y, scene1.vertex.z].';
%scene1_vn = [scene1.vertex.nx, scene1.vertex.ny, scene1.vertex.nz];


%scene1_pts = roty(1)*rotz(2)*[scene1_pts; ones(1, size(scene1_pts,2))];
scene1_pts = scene1_pts(1:3,:).';
scene1_tri = scene1_tri.';


TR = triangulation(scene1_tri, scene1_pts);
scene1_vn = vertexNormal(TR);

disp('Scene Model Loaded and Normals Computed');

% trisurf(TR,'FaceAlpha', 0.8, 'EdgeColor', 'none');
axis equal
hold on

quiver3(chair1_pts(1:model_N:end,1),chair1_pts(1:model_N:end,2),chair1_pts(1:model_N:end,3),...
     chair1_vn(1:model_N:end,1),chair1_vn(1:model_N:end,2),chair1_vn(1:model_N:end,3),0.5,'color','b');
hold off

figure;
chair1_pts_down = chair1_pts(1:model_N:end,:);
chair1_vn_down = chair1_vn(1:model_N:end,:);
scene1_pts_down = scene1_pts(1:scene_N:end,:);
scene1_vn_down = scene1_vn(1:scene_N:end,:);

disp('Downsampled');

[chairMap, chair_d_dist, chair_d_angle] = model_description(chair1_pts_down, chair1_vn_down);

disp('Computed chair map');

[max_alpha, transform_Tmg, transform_Tsg, transform_alpha, max_tots, accumulator] = voting_scheme(chairMap, chair1_pts_down, chair1_vn_down, scene1_pts_down, ...
                                       scene1_vn_down, chair_d_dist, ...
                                                  chair_d_angle);

% size(transform_Tsg)
% size(transform_alpha)
% size(transform_Tmg)
% transform_Tmg

% size(transform_Tmg, 3)

% size(transform_Tsg)
% size(transform_alpha)
% size(transform_Tmg)

% transform_Tsg

%for ii = 1:size(transform_Tmg, 3)
for ii = 1:size(transforms, 3)
    %transform = invht(transform_Tsg(:,:,ii))*transform_alpha(:,:,ii)*transform_Tmg(:,:,ii);
    transform = transforms(:,:,ii);

    if det(transform) > 0
        ii
        transform
        det(transform)

        chair_new_pts = transform*[chair1_pts ones(size(chair1_pts,1),1)].';
        chair_new_pts = chair_new_pts(1:3,:).';

        hold on;
        %scatter3(chair1_pts(:,1),chair1_pts(:,2),chair1_pts(:,3), 1, 'MarkerFaceColor', 'Red')
        %scatter3(scene1_pts(1:10:end,1),scene1_pts(1:10:end,2),scene1_pts(1:10:end,3), 1, 'MarkerFaceColor', 'Blue')
        %h = scatter3(chair_new_pts(:,1),chair_new_pts(:,2),chair_new_pts(:,3), 1, 'MarkerFaceColor', 'Green')
        trisurf(chair1_tri,chair1_pts(:,1),chair1_pts(:,2),chair1_pts(:,3), ...
                'EdgeColor', 'green');
        trisurf(scene1_tri,scene1_pts(:,1),scene1_pts(:,2),scene1_pts(:,3), ...
                'EdgeColor', 'none', 'FaceAlpha', 0.8);
        h = trisurf(chair1_tri,chair_new_pts(:,1),chair_new_pts(:,2),chair_new_pts(:,3), 'EdgeColor', 'yellow');
        cameratoolbar;
        axis equal;
        xlabel('X Axis');
        ylabel('Y Axis');
        zlabel('Z Axis');
        hold off;
        waitfor(h);
    end
end
