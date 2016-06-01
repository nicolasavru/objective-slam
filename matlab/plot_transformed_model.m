function plot_transformed_model(model_ply, transformation, scene_ply)
    [tri_model, pts_model, data_model, ~] = ply_read(model_ply, 'tri');
    [tri_scene, pts_scene, data_scene, ~] = ply_read(scene_ply, 'tri');
    
    model_N = 10;
    scene_N = 10;

    pts_model_down = pts_model(:,1:model_N:end);
    pts_scene_down = pts_scene(:,1:scene_N:end);
    pts_model_down = pts_model_down.';
    pts_scene_down = pts_scene_down.';
    

    model_new_pts = transformation*[pts_model_down ones(size(pts_model_down,1),1)].';
    model_new_pts = model_new_pts(1:3,:).';

    hold on;
    scatter3(pts_model_down(:,1), pts_model_down(:,2), pts_model_down(:,3), 1, 'MarkerFaceColor', 'red', 'MarkerEdgeColor', 'none')
    scatter3(pts_scene_down(:,1), pts_scene_down(:,2), pts_scene_down(:,3), 1, 'MarkerFaceColor', 'blue', 'MarkerEdgeColor', 'none')
    scatter3(model_new_pts(:,1), model_new_pts(:,2), model_new_pts(:,3), 1, 'MarkerFaceColor', 'green', 'MarkerEdgeColor', 'none')

    cameratoolbar;
    axis equal;
    xlabel('X Axis');
    ylabel('Y Axis');
    zlabel('Z Axis');
end