function obj = regSVM(X,Y,varargin)

    [IsOptimizing, RemainingArgs] = classreg.learning.paramoptim.parseOptimizationArgs(varargin);
    if IsOptimizing
        obj = classreg.learning.paramoptim.fitoptimizing('fitrsvm',X,Y,varargin{:});
    else
        obj = RegressionSVM.fit(X,Y,RemainingArgs{:});
    end
end
