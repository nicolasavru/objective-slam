% TRANSLATIONG VECTOR PROCESSING CHECK
% I'm not the best at intuiting non-linear transforms, so here's a little
% utility for checking our logic regards translation vectors and the
% process by which we output a transformation matrix that is the average of
% the transformation associated with a bunch of similar translation
% vectors.

clear all; close all; clc;

% assuming you're in the directory containing the relevant .ply file

% read in a center the model
[chair1_tri,chair1_pts] = ply_read('chair1.ply','tri');
chair1_pts = chair1_pts';
chair1_tri = chair1_tri';

chair1_pts = chair1_pts - repmat(mean(chair1_pts),[size(chair1_pts,1) 1]);

TR = triangulation(chair1_tri, chair1_pts);
chair1_vn = vertexNormal(TR);

% downsample
Nds        = 5000;
ds         = floor(linspace(1,size(chair1_pts,1),Nds));
chair1_pts = chair1_pts(ds,:);
chair1_vn  = chair1_vn(ds,:);

% create a random transform and apply it to the model to create scene
tmp       = 2*pi*rand(3,1) - pi;
T         = rotx(tmp(1))*roty(tmp(2))*rotz(tmp(3));
T(1:3,4)  = randn(3,1);
scene_pts = T*[chair1_pts ones(size(chair1_pts,1),1)]';
scene_pts = scene_pts(1:3,:)';
scene_vn  = chair1_pts + chair1_vn;
scene_vn  = T*[scene_vn ones(size(scene_vn,1),1)]';
scene_vn  = scene_vn(1:3,:)' - scene_pts;

figure
plot3(chair1_pts(:,1),chair1_pts(:,2),chair1_pts(:,3),'.k'), hold on
plot3(scene_pts(:,1),scene_pts(:,2),scene_pts(:,3),'.r')
axis square
cameratoolbar

% now let's pick some matching pairs
Npairs = 100;
% 2 case: 1) Npairs scene reference points 2) one reference point
% UNCOMMENT CASE OF INTEREST
% 1) Npairs scene reference points
% pairs  = round(size(scene_pts,1)*rand(Npairs,2));
% 2) 1 scene reference point
pairs  = round([size(scene_pts,1)*rand*ones(Npairs,1) size(scene_pts,1)*rand(Npairs,1)]);

pairs(logical(pairs(:,1) == pairs(:,2))) = [];

plot3(chair1_pts(pairs(:,1),1),chair1_pts(pairs(:,1),2),chair1_pts(pairs(:,1),3),'.c')
plot3(chair1_pts(pairs(:,2),1),chair1_pts(pairs(:,2),2),chair1_pts(pairs(:,2),3),'.g')

plot3(scene_pts(pairs(:,1),1),scene_pts(pairs(:,1),2),scene_pts(pairs(:,1),3),'.c')
plot3(scene_pts(pairs(:,2),1),scene_pts(pairs(:,2),2),scene_pts(pairs(:,2),3),'.g')

% check consistency of transformation for pairs between model and scene
check_alpha = zeros(Npairs,1);
for ii = 1:Npairs
    fprintf(1,'Pair: %d\n',ii);
    [ T_m_g, T_s_g, alpha ] = trans_model_scene(chair1_pts(pairs(ii,1),:), ...
                                                chair1_vn(pairs(ii,1),:), ...
                                                chair1_pts(pairs(ii,2),:), ...
                                                scene_pts(pairs(ii,1),:), ...
                                                scene_vn(pairs(ii,1),:), ...
                                                scene_pts(pairs(ii,2),:));
    T_new = T_s_g\rotx(alpha)*T_m_g;
    T
    T_new
    alpha
    [T_m_g T_s_g]
    check_alpha(ii) = alpha;
    
    pause
end
bins  = linspace(-pi,pi,100);
tmp   = histc(check_alpha,bins);
figure, bar(bins,tmp)