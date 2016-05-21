% Compute Normals for a PLY file.
function compute_normals( input_file, output_file)
    [tri, pts, data, ~] = ply_read(input_file,'tri');
    TR = triangulation(tri.', pts.');
    vn = vertexNormal(TR);
    data.vertex.nx = vn(:,1);
    data.vertex.ny = vn(:,2);
    data.vertex.nz = vn(:,3);
    data = rmfield(data, 'face');
    
    data.vertex.x = data.vertex.x + abs(min(data.vertex.x)) + 1;
    data.vertex.y = data.vertex.y + abs(min(data.vertex.y)) + 1;
    data.vertex.z = data.vertex.z + abs(min(data.vertex.z)) + 1;
    
    ply_write(data, output_file, 'ascii');
end
