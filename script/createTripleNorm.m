function [fitresult, gof] = createTripleNorm(x_value, y_value)
%CREATEFIT1(X_VALUE,Y_VALUE)
%  Create a fit.
%
%  Data for 'untitled fit 1' fit:
%      X Input : x_value
%      Y Output: y_value
%  Output:
%      fitresult : a fit object representing the fit.
%      gof : structure with goodness-of fit info.

%  General model Gauss3:
%      f(x) = 
%               a1*exp(-((x-b1)/c1)^2) + a2*exp(-((x-b2)/c2)^2) + 
%               a3*exp(-((x-b3)/c3)^2) 

%  See also FIT, CFIT, SFIT.

%  Auto-generated by MATLAB on 05-Aug-2019 17:21:45


%% Fit: 'untitled fit 1'.
[xData, yData] = prepareCurveData( x_value, y_value );

% Set up fittype and options.
ft = fittype( 'gauss3' );
opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
opts.Display = 'Off';
opts.Lower = [-Inf -Inf 0 -Inf -Inf 0 -Inf -Inf 0];
opts.StartPoint = [5622 135.25 3.62220977088804 3588.36886675299 141.25 4.44627624432622 3005.7938110802 130.25 6.27560210275553];

% Fit model to data.
[fitresult, gof] = fit( xData, yData, ft, opts );

% Plot fit with data.
figure( 'Name', 'untitled fit 1' );
h = plot( fitresult, xData, yData );
legend( h, 'y_value vs. x_value', 'untitled fit 1', 'Location', 'NorthEast', 'Interpreter', 'none' );
% Label axes
xlabel( 'x_value', 'Interpreter', 'none' );
ylabel( 'y_value', 'Interpreter', 'none' );
grid on

