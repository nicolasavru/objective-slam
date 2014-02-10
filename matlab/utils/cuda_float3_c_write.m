function [x y] = cuda_float3_c_write (dir)
    % Practicing efficient reads into C\C++ code
    N     = 1000;
    x     = randn(3,N);
    y     = randn(3,N);
    x     = cast(x,'single');
    y     = cast(y,'single');
    fid1   = fopen(sprintf('%s%spoints.txt',dir,filesep),'w+');
    fprintf(fid1,'%10.6f',x);
    fclose(fid1);
    fid2   = fopen(sprintf('%s%snorms.txt',dir,filesep),'w+');
    fprintf(fid2,'%10.6f',y);
    fclose(fid2);
end
