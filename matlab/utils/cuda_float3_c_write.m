function [x y] = cuda_float3_c_write (dir, N)
    % Practicing efficient reads into C\C++ code
    x     = randn(3,N);
    y     = randn(3,N);
    x     = cast(x, 'single');
    y     = cast(y, 'single');
    xu8     = typecast(x(:), 'uint8');
    yu8     = typecast(y(:), 'uint8');
    fid1   = fopen(sprintf('%s%spoints.txt',dir,filesep),'wb', 'ieee-le');
    %fprintf(fid1,'%10.6f',x);
    fwrite(fid1, xu8);
    fclose(fid1);
    fid2   = fopen(sprintf('%s%snorms.txt',dir,filesep),'wb', 'ieee-le');
    %fprintf(fid2,'%10.6f',y);
    fwrite(fid2, yu8);
    fclose(fid2);
end
