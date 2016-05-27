function plot_transformed_model(model_ply, transformation, scene_ply)
    [tri_model, pts_model, data_model, ~] = ply_read(model_ply, 'tri');
    [tri_scene, pts_scene, data_scene, ~] = ply_read(scene_ply, 'tri');
    
    model_N = 10000;
    scene_N = 10000;
    
    pts_model_down = pts_model(1:model_N:end,:);
    pts_scene_down = pts_scene(1:scene_N:end,:);
    pts_model_down = pts_model.';
    pts_scene_down = pts_scene.';

    chair_new_pts = transformation*[pts_model_down ones(size(pts_model_down,1),1)].';
    chair_new_pts = chair_new_pts(1:3,:).';

    hold on;
    scatter3(pts_model_down(:,1), pts_model_down(:,2), pts_model_down(:,3), 1, 'MarkerFaceColor', 'Red')
    scatter3(pts_scene_down(:,1), pts_scene_down(:,2), pts_scene_down(:,3), 1, 'MarkerFaceColor', 'Blue')
    h = scatter3(chair_new_pts(:,1),chair_new_pts(:,2),chair_new_pts(:,3), 1, 'MarkerFaceColor', 'Green')
    %trisurf(tri_model,pts_model_down(:,1), pts_model_down(:,2), pts_model_down(:,3), 'EdgeColor', 'green');
    %trisurf(tri_scene,pts_scene_down(:,1), pts_scene_down(:,2), pts_scene_down(:,3), 'EdgeColor', 'none', 'FaceAlpha', 0.8);
    %h = trisurf(tri_model,chair_new_pts(:,1),chair_new_pts(:,2),chair_new_pts(:,3), 'EdgeColor', 'yellow');
    cameratoolbar;
    axis equal;
    xlabel('X Axis');
    ylabel('Y Axis');
    zlabel('Z Axis');
    %hold off;
    %waitfor(h);
end