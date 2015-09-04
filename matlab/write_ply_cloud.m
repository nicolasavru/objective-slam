function [] = write_ply_cloud(infile, downsample_factor, outfile, centered)

if nargin < 4
  centered = false;
end

[object_tri,object_pts] = ply_read(infile,'tri');
object_pts = object_pts.';
object_tri = object_tri.';

TR = triangulation(object_tri, object_pts);
object_vn = vertexNormal(TR);

if centered
  object_pts = object_pts - repmat(mean(object_pts), size(object_pts,1),1);
end
% object_vn = object_vn - repmat(mean(object_pts), size(object_pts,1),1);
% object_vn = object_vn - object_pts;


disp('Chair Model Loaded and Normals Computed');

object_pts_down = object_pts(1:downsample_factor:end,:);
object_vn_down = object_vn(1:downsample_factor:end,:);

axis equal
hold on
quiver3(object_pts_down(:,1),object_pts_down(:,2),object_pts_down(:,3),...
     object_vn_down(:,1),object_vn_down(:,2),object_vn_down(:,3),0.5,'color','b');
scatter3(object_pts_down(:,1),object_pts_down(:,2),object_pts_down(:,3),'r');
cameratoolbar;
hold off

disp('Downsampled');

num_pts = size(object_pts_down,1);
Data.vertex.x = zeros(num_pts,1);
Data.vertex.y = zeros(num_pts,1);
Data.vertex.z = zeros(num_pts,1);
Data.vertex.nx = zeros(num_pts,1);
Data.vertex.ny = zeros(num_pts,1);
Data.vertex.nz = zeros(num_pts,1);

for ii=1:num_pts
  Data.vertex.x(ii) = object_pts_down(ii,1);
  Data.vertex.y(ii) = object_pts_down(ii,2);
  Data.vertex.z(ii) = object_pts_down(ii,3);
  Data.vertex.nx(ii) = object_vn_down(ii,1);
  Data.vertex.ny(ii) = object_vn_down(ii,2);
  Data.vertex.nz(ii) = object_vn_down(ii,3);
end

ply_write(Data, outfile, 'ascii')

end

