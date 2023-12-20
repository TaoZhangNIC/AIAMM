%% Author : TAO ZHANG  * zt1996nic@gmail.com *
% Created Time : 2023-05-11 08:58
% Last Revised : TAO ZHANG ,2023-07-01
% Remark : Different equation of Parametric oscillator:  \ddot{x} + C(x)\dot{x} + K(x)x=0
%          Case1: Van der Pol 
%             C(x) = -\gamma + \mu x^2, K(x) = 1.  (\gamma=2, \mu=2, x_0=0,\dot{x}_0=1)
%          Case2: Duffing
%             C(x) = \delta, K(x) = k + \epsilon x^2.   (k=1,\delta=0.1,\epsilon=5,x_0=1,\dot{x}_0=0) 
%          Case3: Mathieu
%             C(x) = \xi, K(x) = \alpha + \beta sin(2*t) + \gamma x^2.   (\xi=0.25,\alpha=-1,\beta=-4.6,\gamma=1) 
%          U: Forced or Unforced

function yprime = odeParaOsc(t, y, systemName)
    yprime = zeros(2, 1);

    switch systemName
        case 'van der Pol'
            gamma_vdp = 2;
            mu_vdp = 2;
            CC = -gamma_vdp + mu_vdp * y(1)^2;
            KK = 1;
        case 'Duffing'
            delta_duffing = 0.1;
            k_duffing = 1;
            epsilon_duffing = 5;
            CC = delta_duffing;
            KK = k_duffing + epsilon_duffing * y(1)^2;
        case 'Mathieu'
            c_mathieu = 0.25;
            alpha_mathieu = -1.0;
            beta_mathieu = -4.6;
            gamma_mathieu = 1.0;  
            CC = c_mathieu;
            KK = alpha_mathieu + beta_mathieu * sin(2*t) + gamma_mathieu*y(1)^2;
        otherwise
            error('Invalid systemName');
    end

    yprime(1) = y(2);
    yprime(2) = -CC * y(2) - KK * y(1);
end
