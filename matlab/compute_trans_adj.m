% Compute Normals for a PLY file.
function trans_adj = compute_trans_adj(path)
    files = dir(strcat(path, '*.ply'));
    trans_adj = zeros(3, 1);
    
    for file = files'
        [tri, pts, data, ~] = ply_read(file.name, 'tri');
        trans_adj_x = abs(min(data.vertex.x)) + 1;
        trans_adj_y = abs(min(data.vertex.y)) + 1;
        trans_adj_z = abs(min(data.vertex.z)) + 1;
        
        trans_adj(1, 1) = max(trans_adj(1, 1), trans_adj_x);
        trans_adj(2, 1) = max(trans_adj(2, 1), trans_adj_y);
        trans_adj(3, 1) = max(trans_adj(3, 1), trans_adj_z);
    end
end
