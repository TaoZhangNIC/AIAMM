%% Author : TAO ZHANG  * zt1996nic@gmail.com *
% Created Time : 2023-05-11 08:58
% Last Revised : TAO ZHANG ,2023-07-01
% Remark : Different equation of autonomous chaotic system:
%          Case1: Lorenz 63 system
%          Case2: R{\"o}ssler system
%          Case3: Mean field model
%          Case4: Moore-Spiegel system
%          Case5: Belousov-Zhabotinsky reaction  r \in [0,4]
%          Case6: jerk circuit system
%          Case7: Chua circuit system
%          Case8: 5D autonomous disc dynamo

function dy = odeAutoChaos(t, y, systemName)
% Model setting
switch systemName
    case 'Lorenz system'
        dy = zeros(3, 1);
        dy(1) = 10*(y(2) - y(1));
        dy(2) = y(1)*(28 - y(3)) - y(2);
        dy(3) = y(1)*y(2) - (8/3)*y(3);
    case 'Rossler system'
        dy = zeros(3, 1);
        dy(1) = -(y(2) + y(3));
        dy(2) = y(1) + 0.2*y(2);
        dy(3) = 0.2 + y(3)*(y(1) - 5.7);
    case 'Mean field model'
        dy = zeros(3, 1);
        dy(1) = 2*y(1) - 15*y(2) + (-0.1)*y(1)*y(2);
        dy(2) = 15*y(1) + 2*y(2) + (-0.1)*y(2)*y(3);
        dy(3) = -10*(y(3) - y(1)^2 - y(2)^2);
    case 'Moore-Spiegel system'
        dy = zeros(3, 1);
        dy(1) = y(2);
        dy(2) = -y(2) + 70*y(1) - 40*(y(1) + y(3)) - 70*y(1)*y(3)^2;
        dy(3) = y(1);
    case 'Belousov-Zhabotinsky reaction'
        dy = zeros(3, 1);
        e=0.05;q=0.01;h=0.9;p=3.0;r=1.0;
        dy(1) = (y(1) + y(2) - y(1)*y(2) - q*y(1)^2)/e;
        dy(2) = - y(2) - y(1)*y(2) + 2*h*y(3);
        dy(3) = (y(1) - y(3) - r*y(3))/p;
    case 'Jerk circuit'
        dy = zeros(3, 1);
        a=0.6;
        dy(1) = y(2);
        dy(2) = y(3);
        dy(3) = abs(y(1)) - y(2) - a*y(3) - 1;
    case 'Chua circuit'
        dy = zeros(3, 1);
        p=10;q=14.87;m0=-0.68;m1=-1.27;
        dy(1) = p*(-y(1) + y(2) - (m0*y(1) + 0.5*(m1-m0)*(abs(y(1)+1) - abs(y(1)-1))));
        dy(2) = y(1) - y(2) + y(3);
        dy(3) = -q*y(2);
    case '5D autonomous disc dynamo'
        dy = zeros(5, 1);
        r=8;m=0.2;k1=34;k2=12;g=140.6;
        dy(1) = r*(y(2) - y(1)) + y(4);
        dy(2) = -(1+m)*y(2) + y(1)*y(3) - y(5);
        dy(3) = g*(1 + m*y(1)^2 - (1+m)*y(1)*y(2));
        dy(4) = 2*(1+m)*y(4) + y(1)*y(3) - k1*y(1);
        dy(5) = -m*y(5) + k2*y(2);   
    otherwise
        error('Invalid systemName');
end

end