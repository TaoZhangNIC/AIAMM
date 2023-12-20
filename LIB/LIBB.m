%% Author : TAO ZHANG  * zt1996nic@gmail.com *
% Created Time : 2023-05-11 08:58
% Last Revised : TAO ZHANG ,2023-07-01
% Remark : Library of Mathieu equation


function [Theta,Sym]=LIBB(X,X_OrderMax,Trig_OrderMax,nonsmooth_OrderMax,time)

[~,N_X]=size(X);
Sym_X=sym('x',[N_X,1]);
Sym_sin=sym('sin');
Sym_cos=sym('cos');
Sym_sign=sym('sign');
Sym_time=sym('t');
Theta=[];
%% representation
Index=1;
% Theta(:,Index)=ones(DataN,1);
% Sym{1,Index}=1;

% Index=Index+1;
Theta(:,Index)=sin(2*time).*X(:,1);
Sym{1,Index}=Sym_sin*2*Sym_time*Sym_X(1,1);

Index=Index+1;
Theta(:,Index)=sin(2*time).*X(:,2);
Sym{1,Index}=Sym_sin*2*Sym_time*Sym_X(2,1);

Index=Index+1;
Theta(:,Index)=cos(2*time).*X(:,1);
Sym{1,Index}=Sym_cos*2*Sym_time*Sym_X(1,1);

Index=Index+1;
Theta(:,Index)=cos(2*time).*X(:,2);
Sym{1,Index}=Sym_cos*2*Sym_time*Sym_X(2,1);

Index=Index+1;
Theta(:,Index)=sin(time).*X(:,1);
Sym{1,Index}=Sym_sin*Sym_time*Sym_X(1,1);

Index=Index+1;
Theta(:,Index)=sin(time).*X(:,2);
Sym{1,Index}=Sym_sin*Sym_time*Sym_X(2,1);

Index=Index+1;
Theta(:,Index)=cos(time).*X(:,1);
Sym{1,Index}=Sym_cos*Sym_time*Sym_X(1,1);

Index=Index+1;
Theta(:,Index)=cos(time).*X(:,2);
Sym{1,Index}=Sym_cos*Sym_time*Sym_X(2,1);

%% polynomial representation
if X_OrderMax>=1
    for i=1:N_X
        Index=Index+1;
        Theta(:,Index)=X(:,i);
        Sym{1,Index}=Sym_X(i,1);
    end
end

if X_OrderMax>=2
    for i=1:N_X
        for j=i:N_X
            Index=Index+1;
            Theta(:,Index)=X(:,i).*X(:,j);
            Sym{1,Index}=Sym_X(i,1)*Sym_X(j,1);
        end
    end
end

if X_OrderMax>=3
    for i=1:N_X
        for j=i:N_X
            for k=j:N_X
                Index=Index+1;
                Theta(:,Index)=X(:,i).*X(:,j).*X(:,k);
                Sym{1,Index}=Sym_X(i,1)*Sym_X(j,1)*Sym_X(k,1);
            end
        end
    end
end

if X_OrderMax>=4
    for i=1:N_X
        for j=i:N_X
            for k=j:N_X
                for m=k:N_X
                    Index=Index+1;
                    Theta(:,Index)=X(:,i).*X(:,j).*X(:,k).*X(:,m);
                    Sym{1,Index}=Sym_X(i,1)*Sym_X(j,1)*Sym_X(k,1)*Sym_X(m,1);
                end
            end
        end
    end
end

if X_OrderMax>=5
    for i=1:N_X
        for j=i:N_X
            for k=j:N_X
                for m=k:N_X
                    for p=m:N_X
                        Index=Index+1;
                        Theta(:,Index)=X(:,i).*X(:,j).*X(:,k).*X(:,m).*X(:,p);
                        Sym{1,Index}=Sym_X(i,1)*Sym_X(j,1)*Sym_X(k,1)*Sym_X(m,1)*Sym_X(p,1);
                    end
                end
            end
        end
    end
end


%% trigonometric representation
% sinx cosx
if Trig_OrderMax>=1
    for i=1:N_X
        Index=Index+1;
        Theta(:,Index)=sin(X(:,i));
        Sym{1,Index}=Sym_sin*Sym_X(i,1);
    end
    for i=1:N_X
        Index=Index+1;
        Theta(:,Index)=cos(X(:,i));
        Sym{1,Index}=Sym_cos*Sym_X(i,1);
    end
end

%% non-smooth representation
% sign 
if nonsmooth_OrderMax>=1
    for i=1:N_X
        Index=Index+1;
        Theta(:,Index)=sign(X(:,i));
        Sym{1,Index}=Sym_sign*Sym_X(i,1);
    end
end

