%% This file contains a MATLAB reproduction of some of the python-based
%% experiments in this repository.

%% Author: Milan Mihajlovic (http://www.cs.man.ac.uk/~milan/)





%   Adaptive BDF2, TR and IMR integrators for the LLG problem
%   dm/dt=-a*(m x h)- b*m x [m x h],  m(0)=m0. 
%   h=h_ex+h_mc
%----------------------------------------------------------------
alpha=input('Input the damping parameter alpha:  '); 
field=input('Magnetic field: 1 - h=h_ex, 2 - h=h_ex+h_mc:   ');
if(field==2)
   k1=input('Input the magnetocrystaline parameter k1:   ');
else
   k1=0; 
end
t_end=input('Input the end-point of the interval [0,t_end]:  ');
scheme=input('Input the integration scheme: 1 - BDF2, 2 - TR, 3 - IMR:  ');
mode=input('Input the integration type: 1 - constant, 2 - adaptive:  ');
switch(mode)
   case(1)
       dtc=input('Input the step size dt:  ');
    case(2) 
       if(scheme==3)
          fprintf('Predictor:\n   1 - eBDF3\n');
          fprintf('   2 - AB3\n');
          fprintf('   3 - RK3\n');
          predict=input('   4 - AB2:  ');
          switch(predict)
             case(1)   %   eBDF3
                fprintf('Steps:\n   1 - One-step\n');
                fprintf('   2 - Two-step\n');  
                n_step=input('   3 - One/Two-step:   ');
                if(n_step==2 || n_step==3)
                   fprintf('First step:\n   1 - Absolute stability\n');
                   fprintf('   2 - Border stability\n'); 
                   fprintf('   3 - Min(half step,absolute stability)\n');
                   stab=input('   4 - Half step:   ');
                end
             case(2)   %   AB3
                
             case(3)   %   RK3
                
             case(4)   %   AB2    
                  
             otherwise
                fprintf('Wrong predictor mode\n'); return;
          end
       end
       dt0=input('Input the initial step size dt(0):  ');
       eps_t=input('Input the LTE tolerance:  ');
    otherwise
      fprintf('Wrong integration mode\n'); return;
end
fid=fopen('trace.txt','w');
%   Initial conditions
mx=[0.01/sqrt(1.0001)];        %  x-component of m
my=[0];                        %  y-component of m
mz=[1/sqrt(1.0001)];           %  z-component of m
mx_pred=[0.01/sqrt(1.0001)];   %  predictor history of mx
my_pred=[0];                   %  predictor history of my
mz_pred=[1/sqrt(1.0001)];      %  predictor history of mz
%   External fiels h_ex
hx=[0];         %  x-component of h_ex
hy=[0];         %  y-component of h_ex
hz=[-1.1];      %  z-component of h_ex
%   Magnetocrystaline direction vector
if(field==2)
   ex=1;
   ey=-0.3;
   ez=0;
   e_norm=sqrt(ex^2+ey^2+ez^2);
   ex=ex/e_norm; ey=ey/e_norm; ez=ez/e_norm;
   me=mx(1)*ex+my(1)*ey+mz(1)*ez;
end
%   Mangnetocrystaline filed
if(field==2)
   hmcx=[k1*me*ex];
   hmcy=[k1*me*ey];
   hmcz=[k1*me*ez];
end
switch(scheme)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                  GL-BDF2                                %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                                                                     
    case(1)   %  GL-BDF2
       switch(mode)
            case(1)   %   fixed step size
               t_c=dtc;   %   current time
               n=0;       %   step counter
               dt=[dtc];  %   time step history
               th=[0];    %   discrete time history
               %  start-up with 1 TR step
               n=n+1;
               fprintf('STEP %5i, t=%8.3e, dt=%8.3e\n',n,t_c,dtc);
               fprintf(fid,'STEP %5i, t=%8.3e, dt=%8.3e\n',n,t_c,dtc);
               mx_n=mx(n); my_n=my(n); mz_n=mz(n);   %   Newton init. guess
               m_n=[mx_n;my_n;mz_n];
               hx_n=0; hy_n=0; hz_n=-1.1; 
               hx=[hx,hx_n]; hy=[hy,hy_n]; hz=[hz,hz_n];
               switch(field)
                  case(1) 
                     rx=mx_n+dtc/2*(1/(1+alpha^2)*(my_n*hz(n+1)-mz_n*hy(n+1))+alpha/(1+alpha^2)*...
                             (mx_n*my_n*hy(n+1)+mx_n*mz_n*hz(n+1)-(my_n^2+mz_n^2)*hx(n+1)))-mx(n)+...
                             +dtc/2*(1/(1+alpha^2)*(my(n)*hz(n)-mz(n)*hy(n))+alpha/(1+alpha^2)*...
                             (mx(n)*my(n)*hy(n)+mx(n)*mz(n)*hz(n)-(my(n)^2+mz(n)^2)*hx(n)));
                     ry=my_n+dtc/2*(1/(1+alpha^2)*(mz_n*hx(n+1)-mx_n*hz(n+1))+alpha/(1+alpha^2)*...
                             (my_n*mx_n*hx(n+1)+my_n*mz_n*hz(n+1)-(mx_n^2+mz_n^2)*hy(n+1)))-my(n)+...
                             +dtc/2*(1/(1+alpha^2)*(mz(n)*hx(n)-mx(n)*hz(n))+alpha/(1+alpha^2)*...
                             (my(n)*mx(n)*hx(n)+my(n)*mz(n)*hz(n)-(mx(n)^2+mz(n)^2)*hy(n)));
                     rz=mz_n+dtc/2*(1/(1+alpha^2)*(mx_n*hy(n+1)-my_n*hx(n+1))+alpha/(1+alpha^2)*...
                             (mz_n*mx_n*hx(n+1)+mz_n*my_n*hy(n+1)-(mx_n^2+my_n^2)*hz(n+1)))-mz(n)+...
                             +dtc/2*(1/(1+alpha^2)*(mx(n)*hy(n)-my(n)*hx(n))+alpha/(1+alpha^2)*...
                             (mz(n)*mx(n)*hx(n)+mz(n)*my(n)*hy(n)-(mx(n)^2+my(n)^2)*hz(n))); 
                  case(2)
                     me_n=mx_n*ex+my_n*ey+mz_n*ez;
                     hhx_n=hx_n+k1*me_n*ex; hhy_n=hy_n+k1*me_n*ey; hhz_n=hz_n+k1*me_n*ez;
                     me_o=mx(n)*ex+my(n)*ey+mz(n)*ez;
                     hhx_o=hx(n)+k1*me_o*ex; hhy_o=hy(n)+k1*me_o*ey; hhz_o=hz(n)+k1*me_o*ez;
                     rx=mx_n+dtc/2*(1/(1+alpha^2)*(my_n*hhz_n-mz_n*hhy_n)+alpha/(1+alpha^2)*...
                             (mx_n*my_n*hhy_n+mx_n*mz_n*hhz_n-(my_n^2+mz_n^2)*hhx_n))-mx(n)+...
                             +dtc/2*(1/(1+alpha^2)*(my(n)*hhz_o-mz(n)*hhy_o)+alpha/(1+alpha^2)*...
                             (mx(n)*my(n)*hhy_o+mx(n)*mz(n)*hhz_o-(my(n)^2+mz(n)^2)*hhx_o));
                     ry=my_n+dtc/2*(1/(1+alpha^2)*(mz_n*hhx_n-mx_n*hhz_n)+alpha/(1+alpha^2)*...
                             (my_n*mx_n*hhx_n+my_n*mz_n*hhz_n-(mx_n^2+mz_n^2)*hhy_n))-my(n)+...
                             +dtc/2*(1/(1+alpha^2)*(mz(n)*hhx_o-mx(n)*hhz_o)+alpha/(1+alpha^2)*...
                             (my(n)*mx(n)*hhx_o+my(n)*mz(n)*hhz_o-(mx(n)^2+mz(n)^2)*hhy_o));
                     rz=mz_n+dtc/2*(1/(1+alpha^2)*(mx_n*hhy_n-my_n*hhx_n)+alpha/(1+alpha^2)*...
                             (mz_n*mx_n*hhx_n+mz_n*my_n*hhy_n-(mx_n^2+my_n^2)*hhz_n))-mz(n)+...
                             +dtc/2*(1/(1+alpha^2)*(mx(n)*hhy_o-my(n)*hhx_o)+alpha/(1+alpha^2)*...
                             (mz(n)*mx(n)*hhx_o+mz(n)*my(n)*hhy_o-(mx(n)^2+my(n)^2)*hhz_o)); 
               end          
               r_vec=[rx;ry;rz]; r0_norm=norm(r_vec,2); n_new=0; r_norm=1;
               fprintf('   ||r_nl(0)||_2=%10.4e\n',r0_norm); 
               while(r_norm>=max(1.e-6*r0_norm,1.2e-15))
                  switch(field)
                     case(1) 
                        j11=1+(alpha*dtc)/(2*(1+alpha^2))*(my_n*hy(n+1)+mz_n*hz(n+1));
                        j12=dtc/(2*(1+alpha^2))*hz(n+1)+(alpha*dtc)/(2*(1+alpha^2))*(mx_n*hy(n+1)-2*my_n*hx(n+1));
                        j13=-dtc/(2*(1+alpha^2))*hy(n+1)+(alpha*dtc)/(2*(1+alpha^2))*(mx_n*hz(n+1)-2*mz_n*hx(n+1));
                        j21=-dtc/(2*(1+alpha^2))*hz(n+1)+(alpha*dtc)/(2*(1+alpha^2))*(my_n*hx(n+1)-2*mx_n*hy(n+1));
                        j22=1+(alpha*dtc)/(2*(1+alpha^2))*(mx_n*hx(n+1)+mz_n*hz(n+1));
                        j23=dtc/(2*(1+alpha^2))*hx(n+1)+(alpha*dtc)/(2*(1+alpha^2))*(my_n*hz(n+1)-2*mz_n*hy(n+1));
                        j31=dtc/(2*(1+alpha^2))*hy(n+1)+(alpha*dtc)/(2*(1+alpha^2))*(mz_n*hx(n+1)-2*mx_n*hz(n+1));
                        j32=-dtc/(2*(1+alpha^2))*hx(n+1)+(alpha*dtc)/(2*(1+alpha^2))*(mz_n*hy(n+1)-2*my_n*hz(n+1));
                        j33=1+(alpha*dtc)/(2*(1+alpha^2))*(mx_n*hx(n+1)+my_n*hy(n+1));
                     case(2)
                        j11=1+dtc/(2*(1+alpha^2))*(k1*(my_n*ex*ez-mz_n*ex*ey))+...
                              (alpha*dtc)/(2*(1+alpha^2))*(my_n*hhy_n+k1*my_n*mx_n*ex*ey+mz_n*hhz_n+k1*mx_n*mz_n*ex*ez-...
                              k1*(my_n^2+mz_n^2)*ex*ex);
                        j12=dtc/(2*(1+alpha^2))*(hhz_n+k1*my_n*ey*ez-k1*mz_n*ey*ey)+...
                            (alpha*dtc)/(2*(1+alpha^2))*(mx_n*hhy_n+k1*my_n*mx_n*ey*ey+k1*mx_n*mz_n*ey*ez-...
                             2*my_n*hhx_n-k1*(my_n^2+mz_n^2)*ey*ex);
                        j13=dtc/(2*(1+alpha^2))*(k1*my_n*ez*ez-hhy_n-k1*mz_n*ez*ey)+...
                            (alpha*dtc)/(2*(1+alpha^2))*(k1*my_n*mx_n*ez*ey+mx_n*hhz_n+k1*mx_n*mz_n*ez*ez-...
                             2*mz_n*hhx_n-k1*(my_n^2+mz_n^2)*ez*ex);
                        j21=dtc/(2*(1+alpha^2))*(k1*mz_n*ex*ex-hhz_n-k1*mx_n*ex*ez)+...
                            (alpha*dtc)/(2*(1+alpha^2))*(my_n*hhx_n+k1*mx_n*my_n*ex*ex+...
                            k1*my_n*mz_n*ex*ez-2*mx_n*hhy_n-k1*(mx_n^2+mz_n^2)*ex*ey);
                        j22=1+dtc/(2*(1+alpha^2))*(k1*(mz_n*ey*ex-mx_n*ey*ez))+...
                              (alpha*dtc)/(2*(1+alpha^2))*(mx_n*hhx_n+k1*mx_n*my_n*ey*ex+mz_n*hhz_n+k1*my_n*mz_n*ey*ez-...
                              k1*(mx_n^2+mz_n^2)*ey*ey);
                        j23=dtc/(2*(1+alpha^2))*(hhx_n+k1*mz_n*ez*ex-k1*mx_n*ez*ez)+...
                            (alpha*dtc)/(2*(1+alpha^2))*(k1*mx_n*my_n*ez*ex+my_n*hhz_n+k1*my_n*mz_n*ez*ez-...
                            2*mz_n*hhy_n-k1*(mx_n^2+mz_n^2)*ez*ey);
                        j31=dtc/(2*(1+alpha^2))*(hhy_n+k1*mx_n*ex*ey-k1*my_n*ex*ex)+...
                            (alpha*dtc)/(2*(1+alpha^2))*(mz_n*hhx_n+k1*mx_n*mz_n*ex*ex+k1*my_n*mz_n*ex*ey-...
                            2*mx_n*hhz_n-k1*(mx_n^2+my_n^2)*ex*ez);
                        j32=dtc/(2*(1+alpha^2))*(k1*mx_n*ey*ey-hhx_n-k1*my_n*ey*ex)+...
                            (alpha*dtc)/(2*(1+alpha^2))*(k1*mx_n*mz_n*ey*ex+mz_n*hhy_n+k1*my_n*mz_n*ey*ey-...
                            2*my_n*hhz_n-k1*(mx_n^2+my_n^2)*ey*ez);
                        j33=1+dtc/(2*(1+alpha^2))*(k1*(mx_n*ez*ey-my_n*ez*ex))+...
                              (alpha*dtc)/(2*(1+alpha^2))*(mx_n*hhx_n+k1*mx_n*mz_n*ez*ex+my_n*hhy_n+k1*my_n*mz_n*ez*ey-...
                              k1*(mx_n^2+my_n^2)*ez*ez);  
                  end
                  J=[j11,j12,j13;j21,j22,j23;j31,j32,j33];
                  dm_n=J\r_vec;
                  m_n=m_n-dm_n;
                  mx_n=m_n(1); my_n=m_n(2); mz_n=m_n(3);
                  hx_n=0; hy_n=0; hz_n=-1.1; 
                  switch(field)
                     case(1) 
                        rx=mx_n+dtc/2*(1/(1+alpha^2)*(my_n*hz(n+1)-mz_n*hy(n+1))+alpha/(1+alpha^2)*...
                                (mx_n*my_n*hy(n+1)+mx_n*mz_n*hz(n+1)-(my_n^2+mz_n^2)*hx(n+1)))-mx(n)+...
                                +dtc/2*(1/(1+alpha^2)*(my(n)*hz(n)-mz(n)*hy(n))+alpha/(1+alpha^2)*...
                                (mx(n)*my(n)*hy(n)+mx(n)*mz(n)*hz(n)-(my(n)^2+mz(n)^2)*hx(n)));
                        ry=my_n+dtc/2*(1/(1+alpha^2)*(mz_n*hx(n+1)-mx_n*hz(n+1))+alpha/(1+alpha^2)*...
                                (my_n*mx_n*hx(n+1)+my_n*mz_n*hz(n+1)-(mx_n^2+mz_n^2)*hy(n+1)))-my(n)+...
                                +dtc/2*(1/(1+alpha^2)*(mz(n)*hx(n)-mx(n)*hz(n))+alpha/(1+alpha^2)*...
                                (my(n)*mx(n)*hx(n)+my(n)*mz(n)*hz(n)-(mx(n)^2+mz(n)^2)*hy(n)));
                        rz=mz_n+dtc/2*(1/(1+alpha^2)*(mx_n*hy(n+1)-my_n*hx(n+1))+alpha/(1+alpha^2)*...
                                (mz_n*mx_n*hx(n+1)+mz_n*my_n*hy(n+1)-(mx_n^2+my_n^2)*hz(n+1)))-mz(n)+...
                                +dtc/2*(1/(1+alpha^2)*(mx(n)*hy(n)-my(n)*hx(n))+alpha/(1+alpha^2)*...
                                (mz(n)*mx(n)*hx(n)+mz(n)*my(n)*hy(n)-(mx(n)^2+my(n)^2)*hz(n))); 
                     case(2)
                        me_n=mx_n*ex+my_n*ey+mz_n*ez;
                        hhx_n=hx_n+k1*me_n*ex; hhy_n=hy_n+k1*me_n*ey; hhz_n=hz_n+k1*me_n*ez;
                        rx=mx_n+dtc/2*(1/(1+alpha^2)*(my_n*hhz_n-mz_n*hhy_n)+alpha/(1+alpha^2)*...
                                (mx_n*my_n*hhy_n+mx_n*mz_n*hhz_n-(my_n^2+mz_n^2)*hhx_n))-mx(n)+...
                                +dtc/2*(1/(1+alpha^2)*(my(n)*hhz_o-mz(n)*hhy_o)+alpha/(1+alpha^2)*...
                                (mx(n)*my(n)*hhy_o+mx(n)*mz(n)*hhz_o-(my(n)^2+mz(n)^2)*hhx_o));
                        ry=my_n+dtc/2*(1/(1+alpha^2)*(mz_n*hhx_n-mx_n*hhz_n)+alpha/(1+alpha^2)*...
                                (my_n*mx_n*hhx_n+my_n*mz_n*hhz_n-(mx_n^2+mz_n^2)*hhy_n))-my(n)+...
                                +dtc/2*(1/(1+alpha^2)*(mz(n)*hhx_o-mx(n)*hhz_o)+alpha/(1+alpha^2)*...
                                (my(n)*mx(n)*hhx_o+my(n)*mz(n)*hhz_o-(mx(n)^2+mz(n)^2)*hhy_o));
                        rz=mz_n+dtc/2*(1/(1+alpha^2)*(mx_n*hhy_n-my_n*hhx_n)+alpha/(1+alpha^2)*...
                                (mz_n*mx_n*hhx_n+mz_n*my_n*hhy_n-(mx_n^2+my_n^2)*hhz_n))-mz(n)+...
                                +dtc/2*(1/(1+alpha^2)*(mx(n)*hhy_o-my(n)*hhx_o)+alpha/(1+alpha^2)*...
                                (mz(n)*mx(n)*hhx_o+mz(n)*my(n)*hhy_o-(mx(n)^2+my(n)^2)*hhz_o)); 
                  end
                  r_vec=[rx;ry;rz]; r_norm=norm(r_vec,2);
                  n_new=n_new+1;
               end
               fprintf('   Newton method converged in %1i steps\n',n_new);
               fprintf('   ||r_nl(%2i)||_2=%10.4e\n',n_new,r_norm); 
               mx=[mx,mx_n]; my=[my,my_n]; mz=[mz,mz_n];
               if(field==2)
                  me=mx_n*ex+my_n*ey+mz_n*ez;
                  hmcx=[hmcx,k1*me*ex]; hmcy=[hmcy,k1*me*ey]; hmcz=[hmcz,k1*me*ez];
               end
               dt=[dt,dtc]; th=[th,t_c]; t_c=t_c+dtc; fin=0;  
               %   BDF2 loop
               while(t_c<=t_end && fin==0)
                  n=n+1;
                  fprintf('STEP %5i, t=%8.3e, dt=%8.3e\n',n,t_c,dtc);
                  fprintf(fid,'STEP %5i, t=%8.3e, dt=%8.3e\n',n,t_c,dtc);
                  mx_n=mx(n); my_n=my(n); mz_n=mz(n);   %   Newton init. guess
                  m_n=[mx_n;my_n;mz_n];
                  hx_n=0; hy_n=0; hz_n=-1.1; 
                  hx=[hx,hx_n]; hy=[hy,hy_n]; hz=[hz,hz_n];
                  switch(field)
                     case(1)  
                        rx=mx_n+2*dtc/3*(1/(1+alpha^2)*(my_n*hz(n+1)-mz_n*hy(n+1))+alpha/(1+alpha^2)*...
                                        (mx_n*my_n*hy(n+1)+mx_n*mz_n*hz(n+1)-(my_n^2+mz_n^2)*hx(n+1)))-...
                                        4/3*mx(n)+1/3*mx(n-1);
                        ry=my_n+2*dtc/3*(1/(1+alpha^2)*(mz_n*hx(n+1)-mx_n*hz(n+1))+alpha/(1+alpha^2)*...
                                        (my_n*mx_n*hx(n+1)+my_n*mz_n*hz(n+1)-(mx_n^2+mz_n^2)*hy(n+1)))-...
                                        4/3*my(n)+1/3*my(n-1);
                        rz=mz_n+2*dtc/3*(1/(1+alpha^2)*(mx_n*hy(n+1)-my_n*hx(n+1))+alpha/(1+alpha^2)*...
                                        (mz_n*mx_n*hx(n+1)+mz_n*my_n*hy(n+1)-(mx_n^2+my_n^2)*hz(n+1)))-...
                                        4/3*mz(n)+1/3*mz(n-1);
                     case(2)
                        me_n=mx_n*ex+my_n*ey+mz_n*ez;
                        hhx_n=hx_n+k1*me_n*ex; hhy_n=hy_n+k1*me_n*ey; hhz_n=hz_n+k1*me_n*ez;
                        rx=mx_n+2*dtc/3*(1/(1+alpha^2)*(my_n*hhz_n-mz_n*hhy_n)+alpha/(1+alpha^2)*...
                                        (mx_n*my_n*hhy_n+mx_n*mz_n*hhz_n-(my_n^2+mz_n^2)*hhx_n))-...
                                        4/3*mx(n)+1/3*mx(n-1);
                        ry=my_n+2*dtc/3*(1/(1+alpha^2)*(mz_n*hhx_n-mx_n*hhz_n)+alpha/(1+alpha^2)*...
                                        (my_n*mx_n*hhx_n+my_n*mz_n*hhz_n-(mx_n^2+mz_n^2)*hhy_n))-...
                                        4/3*my(n)+1/3*my(n-1);
                        rz=mz_n+2*dtc/3*(1/(1+alpha^2)*(mx_n*hy(n+1)-my_n*hx(n+1))+alpha/(1+alpha^2)*...
                                        (mz_n*mx_n*hhx_n+mz_n*my_n*hhy_n-(mx_n^2+my_n^2)*hhz_n))-...
                                        4/3*mz(n)+1/3*mz(n-1);
                  end
                  r_vec=[rx;ry;rz]; r0_norm=norm(r_vec,2); n_new=0; r_norm=1;
                  fprintf('   ||r_nl(0)||_2=%10.4e\n',r0_norm);
                  while(r_norm>=max(1.e-9*r0_norm,1.2e-15))
                     switch(field)
                        case(1) 
                           j11=1+(2*alpha*dtc)/(3*(1+alpha^2))*(my_n*hy(n+1)+mz_n*hz(n+1));
                           j12=(2*dtc)/(3*(1+alpha^2))*hz(n+1)+(2*alpha*dtc)/(3*(1+alpha^2))*...
                                       (mx_n*hy(n+1)-2*my_n*hx(n+1));
                           j13=-(2*dtc)/(3*(1+alpha^2))*hy(n+1)+(2*alpha*dtc)/(3*(1+alpha^2))*...
                                       (mx_n*hz(n+1)-2*mz_n*hx(n+1));
                           j21=-(2*dtc)/(3*(1+alpha^2))*hz(n+1)+(2*alpha*dtc)/(3*(1+alpha^2))*...
                                       (my_n*hx(n+1)-2*mx_n*hy(n+1));
                           j22=1+(2*alpha*dtc)/(3*(1+alpha^2))*(mx_n*hx(n+1)+mz_n*hz(n+1));
                           j23=(2*dtc)/(3*(1+alpha^2))*hx(n+1)+(2*alpha*dtc)/(3*(1+alpha^2))*...
                                       (my_n*hz(n+1)-2*mz_n*hy(n+1));
                           j31=(2*dtc)/(3*(1+alpha^2))*hy(n+1)+(2*alpha*dtc)/(3*(1+alpha^2))*...
                                       (mz_n*hx(n+1)-2*mx_n*hz(n+1));
                           j32=-(2*dtc)/(3*(1+alpha^2))*hx(n+1)+(2*alpha*dtc)/(3*(1+alpha^2))*...
                                       (mz_n*hy(n+1)-2*my_n*hz(n+1));
                           j33=1+(2*alpha*dtc)/(3*(1+alpha^2))*(mx_n*hx(n+1)+my_n*hy(n+1));     
                        case(2)
                           j11=1+(2*dtc)/(3*(1+alpha^2))*(k1*(my_n*ex*ez-mz_n*ex*ey))+...
                               (2*alpha*dtc)/(3*(1+alpha^2))*(my_n*hhy_n+k1*my_n*mx_n*ex*ey+mz_n*hhz_n+k1*mx_n*mz_n*ex*ez-...
                               k1*(my_n^2+mz_n^2)*ex*ex);
                           j12=(2*dtc)/(3*(1+alpha^2))*(hhz_n+k1*my_n*ey*ez-k1*mz_n*ey*ey)+...
                               (2*alpha*dtc)/(3*(1+alpha^2))*(mx_n*hhy_n+k1*my_n*mx_n*ey*ey+k1*mx_n*mz_n*ey*ez-...
                                2*my_n*hhx_n-k1*(my_n^2+mz_n^2)*ey*ex);
                           j13=(2*dtc)/(3*(1+alpha^2))*(k1*my_n*ez*ez-hhy_n-k1*mz_n*ez*ey)+...
                               (2*alpha*dtc)/(3*(1+alpha^2))*(k1*my_n*mx_n*ez*ey+mx_n*hhz_n+k1*mx_n*mz_n*ez*ez-...
                                2*mz_n*hhx_n-k1*(my_n^2+mz_n^2)*ez*ex);
                           j21=(2*dtc)/(3*(1+alpha^2))*(k1*mz_n*ex*ex-hhz_n-k1*mx_n*ex*ez)+...
                               (2*alpha*dtc)/(3*(1+alpha^2))*(my_n*hhx_n+k1*mx_n*my_n*ex*ex+...
                               k1*my_n*mz_n*ex*ez-2*mx_n*hhy_n-k1*(mx_n^2+mz_n^2)*ex*ey);
                           j22=1+(2*dtc)/(3*(1+alpha^2))*(k1*(mz_n*ey*ex-mx_n*ey*ez))+...
                               (2*alpha*dtc)/(3*(1+alpha^2))*(mx_n*hhx_n+k1*mx_n*my_n*ey*ex+mz_n*hhz_n+k1*my_n*mz_n*ey*ez-...
                               k1*(mx_n^2+mz_n^2)*ey*ey);
                           j23=(2*dtc)/(3*(1+alpha^2))*(hhx_n+k1*mz_n*ez*ex-k1*mx_n*ez*ez)+...
                               (2*alpha*dtc)/(3*(1+alpha^2))*(k1*mx_n*my_n*ez*ex+my_n*hhz_n+k1*my_n*mz_n*ez*ez-...
                               2*mz_n*hhy_n-k1*(mx_n^2+mz_n^2)*ez*ey);
                           j31=(2*dtc)/(3*(1+alpha^2))*(hhy_n+k1*mx_n*ex*ey-k1*my_n*ex*ex)+...
                               (2*alpha*dtc)/(3*(1+alpha^2))*(mz_n*hhx_n+k1*mx_n*mz_n*ex*ex+k1*my_n*mz_n*ex*ey-...
                               2*mx_n*hhz_n-k1*(mx_n^2+my_n^2)*ex*ez);
                           j32=(2*dtc)/(3*(1+alpha^2))*(k1*mx_n*ey*ey-hhx_n-k1*my_n*ey*ex)+...
                               (2*alpha*dtc)/(3*(1+alpha^2))*(k1*mx_n*mz_n*ey*ex+mz_n*hhy_n+k1*my_n*mz_n*ey*ey-...
                               2*my_n*hhz_n-k1*(mx_n^2+my_n^2)*ey*ez);
                           j33=1+(2*dtc)/(3*(1+alpha^2))*(k1*(mx_n*ez*ey-my_n*ez*ex))+...
                              (2*alpha*dtc)/(3*(1+alpha^2))*(mx_n*hhx_n+k1*mx_n*mz_n*ez*ex+my_n*hhy_n+k1*my_n*mz_n*ez*ey-...
                              k1*(mx_n^2+my_n^2)*ez*ez);    
                     end
                     J=[j11,j12,j13;j21,j22,j23;j31,j32,j33];
                     dm_n=J\r_vec;
                     m_n=m_n-dm_n;
                     mx_n=m_n(1); my_n=m_n(2); mz_n=m_n(3);
                     switch(field)
                        case(1) 
                           rx=mx_n+2*dtc/3*(1/(1+alpha^2)*(my_n*hz(n+1)-mz_n*hy(n+1))+alpha/(1+alpha^2)*...
                                           (mx_n*my_n*hy(n+1)+mx_n*mz_n*hz(n+1)-(my_n^2+mz_n^2)*hx(n+1)))-...
                                           4/3*mx(n)+1/3*mx(n-1);
                           ry=my_n+2*dtc/3*(1/(1+alpha^2)*(mz_n*hx(n+1)-mx_n*hz(n+1))+alpha/(1+alpha^2)*...
                                           (my_n*mx_n*hx(n+1)+my_n*mz_n*hz(n+1)-(mx_n^2+mz_n^2)*hy(n+1)))-...
                                           4/3*my(n)+1/3*my(n-1);
                           rz=mz_n+2*dtc/3*(1/(1+alpha^2)*(mx_n*hy(n+1)-my_n*hx(n+1))+alpha/(1+alpha^2)*...
                                           (mz_n*mx_n*hx(n+1)+mz_n*my_n*hy(n+1)-(mx_n^2+my_n^2)*hz(n+1)))-...
                                           4/3*mz(n)+1/3*mz(n-1);
                        case(2)
                           me_n=mx_n*ex+my_n*ey+mz_n*ez;
                           hhx_n=hx_n+k1*me_n*ex; hhy_n=hy_n+k1*me_n*ey; hhz_n=hz_n+k1*me_n*ez;
                           rx=mx_n+2*dtc/3*(1/(1+alpha^2)*(my_n*hhz_n-mz_n*hhy_n)+alpha/(1+alpha^2)*...
                                           (mx_n*my_n*hhy_n+mx_n*mz_n*hhz_n-(my_n^2+mz_n^2)*hhx_n))-...
                                           4/3*mx(n)+1/3*mx(n-1);
                           ry=my_n+2*dtc/3*(1/(1+alpha^2)*(mz_n*hhx_n-mx_n*hhz_n)+alpha/(1+alpha^2)*...
                                           (my_n*mx_n*hhx_n+my_n*mz_n*hhz_n-(mx_n^2+mz_n^2)*hhy_n))-...
                                           4/3*my(n)+1/3*my(n-1);
                           rz=mz_n+2*dtc/3*(1/(1+alpha^2)*(mx_n*hy(n+1)-my_n*hx(n+1))+alpha/(1+alpha^2)*...
                                           (mz_n*mx_n*hhx_n+mz_n*my_n*hhy_n-(mx_n^2+my_n^2)*hhz_n))-...
                                           4/3*mz(n)+1/3*mz(n-1); 
                     end
                     r_vec=[rx;ry;rz]; r_norm=norm(r_vec,2);
                     n_new=n_new+1;
                  end
                  fprintf('   Newton method converged in %1i steps\n',n_new);
                  fprintf('   ||r_nl(%2i)||_2=%10.4e\n',n_new,r_norm); 
                  mx=[mx,mx_n]; my=[my,my_n]; mz=[mz,mz_n];
                  if(field==2)
                     me=mx_n*ex+my_n*ey+mz_n*ez;
                     hmcx=[hmcx,k1*me*ex]; hmcy=[hmcy,k1*me*ey]; hmcz=[hmcz,k1*me*ez];
                  end
                  fprintf('   BDF2:  mx(%8.4f)=%7.4f; my(%8.4f)=%7.4f; mz(%8.4f)=%7.4f\n',t_c,mx_n,t_c,my_n,t_c,mz_n);
                  fprintf(fid,'   BDF2:  mx(%8.4f)=%7.4f; my(%8.4f)=%7.4f; mz(%8.4f)=%7.4f\n',t_c,mx_n,t_c,my_n,t_c,mz_n);
                  if(t_c==t_end)
                     fin=1;
                  end
                  t_c=t_c+dtc;
                  if(t_c>=t_end)
                     dtc=dtc-t_c+t_end; 
                     t_c=t_end;
                     if(dtc<=1.e-6)
                        fin=1;
                     end
                  end
                  dt=[dt,dtc]; th=[th,t_c]; 
               end
               set(gcf,'Position',[575,240,700,700]);
               subplot(221); plot(th,mx,'r.-','MarkerSize',4); 
               axis([0 t_end -1.01*max(abs(mx)) 1.01*max(abs(mx))]);
               title('x-component of the magnetization m');
               xlabel('t'); ylabel('mx');
               subplot(222); plot(th,my,'b.-','MarkerSize',4);
               axis([0 t_end -1.01*max(abs(my)) 1.01*max(abs(my))]);
               title('y-component of the magnetization m');
               xlabel('t'); ylabel('my');
               subplot(223); plot(th,mz,'g.-','MarkerSize',4);
               axis([0 t_end -1.01*max(abs(mz)) 1.01*max(abs(mz))]);
               title('z-component of the magnetization m');
               xlabel('t'); ylabel('mz');
               subplot(224); plot(th,dt,'k.-','MarkerSize',4);
               axis([0 t_end 0 1.1*max(dt)]);
               title('Time step history');
               xlabel('t'); ylabel('dt');
            case(2)   %   variable step size  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
               dt=[dt0]; %  the time step history
               t_c=dt0;  %  current time
               dtc=dt0;  %  current step size
               th=[0];   %  time history
               dh=[0,0,0]; %  LTE estimate history
               %  Startup (1 step of IMR with dt(0))
               for n=1:1 
                  fprintf('STEP %5i, t=%8.3e\n',n,t_c);
                  fprintf(fid,'STEP %5i, t=%8.3e\n',n,t_c);
                  mx_n=mx(n); my_n=my(n); mz_n=mz(n);   %   Newton init. guess
                  m_n=[mx_n;my_n;mz_n];
                  hx_n=0; hy_n=0; hz_n=-1.1;   %  external field
                  hx=[hx,hx_n]; hy=[hy,hy_n]; hz=[hz,hz_n];
                  mx_h=(mx_n+mx(n))/2; my_h=(my_n+my(n))/2; mz_h=(mz_n+mz(n))/2;
                  hx_h=(hx(n+1)+hx(n))/2; hy_h=(hy(n+1)+hy(n))/2; hz_h=(hz(n+1)+hz(n))/2;
                  switch(field)
                     case(1) 
                        rx=mx_n+dtc*(1/(1+alpha^2)*(my_h*hz_h-mz_h*hy_h)+alpha/(1+alpha^2)*...
                                    (my_h*mx_h*hy_h+mx_h*mz_h*hz_h-(my_h^2+mz_h^2)*hx_h))-mx(n);
                        ry=my_n+dtc*(1/(1+alpha^2)*(mz_h*hx_h-mx_h*hz_h)+alpha/(1+alpha^2)*...
                                    (mx_h*my_h*hx_h+my_h*mz_h*hz_h-(mx_h^2+mz_h^2)*hy_h))-my(n);
                        rz=mz_n+dtc*(1/(1+alpha^2)*(mx_h*hy_h-my_h*hx_h)+alpha/(1+alpha^2)*...
                                    (mx_h*mz_h*hx_h+my_h*mz_h*hy_h-(mx_h^2+my_h^2)*hz_h))-mz(n); 
                     case(2)
                        me_h=(mx_n+mx(n))/2*ex+(my_n+my(n))/2*ey+(mz_n+mz(n))/2*ez;
                        hhx_h=hx_h+k1*me_h*ex; hhy_h=hy_h+k1*me_h*ey; hhz_h=hz_h+k1*me_h*ez;
                        rx=mx_n+dtc*(1/(1+alpha^2)*(my_h*hhz_h-mz_h*hhy_h)+alpha/(1+alpha^2)*...
                                    (my_h*mx_h*hhy_h+mx_h*mz_h*hhz_h-(my_h^2+mz_h^2)*hhx_h))-mx(n);
                        ry=my_n+dtc*(1/(1+alpha^2)*(mz_h*hhx_h-mx_h*hhz_h)+alpha/(1+alpha^2)*...
                                    (mx_h*my_h*hhx_h+my_h*mz_h*hhz_h-(mx_h^2+mz_h^2)*hhy_h))-my(n);
                        rz=mz_n+dtc*(1/(1+alpha^2)*(mx_h*hhy_h-my_h*hhx_h)+alpha/(1+alpha^2)*...
                                    (mx_h*mz_h*hhx_h+my_h*mz_h*hhy_h-(mx_h^2+my_h^2)*hhz_h))-mz(n); 
                  end
                  r_vec=[rx;ry;rz]; r0_norm=norm(r_vec,2); n_new=0; r_norm=1;
                  fprintf('   ||r_nl(0)||_2=%10.4e\n',r0_norm);
                  while(r_norm>=max(1.e-9*r0_norm,1.2e-15))
                     switch(field)
                        case(1) 
                           j11=1+(alpha*dtc)/(2*(1+alpha^2))*(my_h*hy_h+mz_h*hz_h);
                           j12=dtc/(2*(1+alpha^2))*hz_h+(alpha*dtc)/(2*(1+alpha^2))*(mx_h*hy_h-2*my_h*hx_h);
                           j13=-dtc/(2*(1+alpha^2))*hy_h+(alpha*dtc)/(2*(1+alpha^2))*(mx_h*hz_h-2*mz_h*hx_h);
                           j21=-dtc/(2*(1+alpha^2))*hz_h+(alpha*dtc)/(2*(1+alpha^2))*(my_h*hx_h-2*mx_h*hy_h);
                           j22=1+(alpha*dtc)/(2*(1+alpha^2))*(mx_h*hx_h+mz_h*hz_h);
                           j23=dtc/(2*(1+alpha^2))*hx_h+(alpha*dtc)/(2*(1+alpha^2))*(my_h*hz_h-2*mz_h*hy_h);
                           j31=dtc/(2*(1+alpha^2))*hy_h+(alpha*dtc)/(2*(1+alpha^2))*(mz_h*hx_h-2*mx_h*hz_h);
                           j32=-dtc/(2*(1+alpha^2))*hx_h+(alpha*dtc)/(2*(1+alpha^2))*(mz_h*hy_h-2*my_h*hz_h);
                           j33=1+(alpha*dtc)/(2*(1+alpha^2))*(mx_h*hx_h+my_h*hy_h);
                        case(2)
                           j11=1+(dtc/(2*(1+alpha^2)))*(k1*(my_h*ex*ez-mz_h*ex*ey))+(dtc*alpha)/(2*(1+alpha^2))*...
                                 (my_h*hhy_h+k1*my_h*mx_h*ex*ey+mz_h*hhz_h+k1*mx_h*mz_h*ex*ez-k1*(my_h^2+mz_h^2)*ex*ex);
                           j12=dtc/(2*(1+alpha^2))*(hhz_h-k1*mz_h*ey*ey+k1*my_h*ey*ez)+(dtc*alpha)/(2*(1+alpha^2))*...
                                 (mx_h*hy_h+k1*mx_h*my_h*ey*ey+k1*mx_h*mz_h*ey*ez-2*my_h*hhx_h-k1*(my_h^2+mz_h^2)*ey*ex);
                           j13=dtc/(2*(1+alpha^2))*(k1*my_h*ez*ez-hhy_h-k1*mz_h*ez*ey)+(dtc*alpha)/(2*(1+alpha^2))*...
                                 (k1*my_h*mx_h*ez*ey+mx_h*hhz_h+k1*mx_h*mz_h*ez*ez-2*mz_h*hhx_h-k1*(my_h^2+mz_h^2)*ez*ex);
                           j21=dtc/(2*(1+alpha^2))*(k1*mz_h*ex*ex-hhz_h-k1*mx_h*ex*ez)+(dtc*alpha)/(2*(1+alpha^2))*...
                                 (my_h*hhx_h+k1*mx_h*my_h*ex*ex+k1*my_h*mz_h*ex*ez-2*mx_h*hhy_h-k1*(mx_h^2+mz_h^2)*ex*ey);
                           j22=1+(dtc/(2*(1+alpha^2)))*(k1*(mz_h*ey*ex-mx_h*ey*ez))+(dtc*alpha)/(2*(1+alpha^2))*...
                                 (mx_h*hx_h+k1*mx_h*my_h*ey*ex+mz_h*hhz_h+k1*my_h*mz_h*ey*ez-k1*(mx_h^2+mz_h^2)*ey*ey);
                           j23=dtc/(2*(1+alpha^2))*(hhx_h+k1*mz_h*ez*ex-k1*mx_h*ez*ez)+(dtc*alpha)/(2*(1+alpha^2))*...
                                 (k1*mx_h*my_h*ez*ex+my_h*hhz_h+k1*my_h*mz_h*ez*ez-2*mz_h*hhy_h-k1*(mx_h^2+mz_h^2)*ez*ey);
                           j31=dtc/(2*(1+alpha^2))*(hhy_h-k1*my_h*ex*ex+k1*mx_h*ex*ey)+(dtc*alpha)/(2*(1+alpha^2))*...
                                 (mz_h*hhx_h+k1*mx_h*mz_h*ex*ex+k1*my_h*mz_h*ex*ey-2*mx_h*hhz_h-k1*(mx_h^2+my_h^2)*ex*ez);
                           j32=dtc/(2*(1+alpha^2))*(k1*mx_h*ey*ey-hhx_h-k1*my_h*ey*ex)+(dtc*alpha)/(2*(1+alpha^2))*...
                                 (k1*mx_h*mz_h*ey*ex+mz_h*hhy_h+k1*my_h*mz_h*ey*ey-2*my_h*hhz_h-k1*(mx_h^2+my_h^2)*ey*ez);
                           j33=1+(dtc/(2*(1+alpha^2)))*(k1*(mx_h*ez*ey-my_h*ez*ex))+(dtc*alpha)/(2*(1+alpha^2))*...
                                 (mx_h*hhx_h+k1*mx_h*mz_h*ez*ex+my_h*hhy_h+k1*my_h*mz_h*ez*ey-k1*(mx_h^2+my_h)^2*ez*ez);
                     end
                     J=[j11,j12,j13;j21,j22,j23;j31,j32,j33];
                     dm_n=J\r_vec;
                     m_n=m_n-dm_n;
                     mx_n=m_n(1); my_n=m_n(2); mz_n=m_n(3);
                     mx_h=(mx_n+mx(n))/2; my_h=(my_n+my(n))/2; mz_h=(mz_n+mz(n))/2;
                     switch(field)
                        case(1) 
                           rx=mx_n+dtc*(1/(1+alpha^2)*(my_h*hz_h-mz_h*hy_h)+alpha/(1+alpha^2)*...
                                       (my_h*mx_h*hy_h+mx_h*mz_h*hz_h-(my_h^2+mz_h^2)*hx_h))-mx(n);
                           ry=my_n+dtc*(1/(1+alpha^2)*(mz_h*hx_h-mx_h*hz_h)+alpha/(1+alpha^2)*...
                                       (mx_h*my_h*hx_h+my_h*mz_h*hz_h-(mx_h^2+mz_h^2)*hy_h))-my(n);
                           rz=mz_n+dtc*(1/(1+alpha^2)*(mx_h*hy_h-my_h*hx_h)+alpha/(1+alpha^2)*...
                                       (mx_h*mz_h*hx_h+my_h*mz_h*hy_h-(mx_h^2+my_h^2)*hz_h))-mz(n);
                        case(2)
                           me_h=(mx_n+mx(n))/2*ex+(my_n+my(n))/2*ey+(mz_n+mz(n))/2*ez;
                           hhx_h=hx_h+k1*me_h*ex; hhy_h=hy_h+k1*me_h*ey; hhz_h=hz_h+k1*me_h*ez;
                           rx=mx_n+dtc*(1/(1+alpha^2)*(my_h*hhz_h-mz_h*hhy_h)+alpha/(1+alpha^2)*...
                                       (my_h*mx_h*hhy_h+mx_h*mz_h*hhz_h-(my_h^2+mz_h^2)*hhx_h))-mx(n);
                           ry=my_n+dtc*(1/(1+alpha^2)*(mz_h*hhx_h-mx_h*hhz_h)+alpha/(1+alpha^2)*...
                                       (mx_h*my_h*hhx_h+my_h*mz_h*hhz_h-(mx_h^2+mz_h^2)*hhy_h))-my(n);
                           rz=mz_n+dtc*(1/(1+alpha^2)*(mx_h*hhy_h-my_h*hhx_h)+alpha/(1+alpha^2)*...
                                       (mx_h*mz_h*hhx_h+my_h*mz_h*hhy_h-(mx_h^2+my_h^2)*hhz_h))-mz(n);    
                     end
                     r_vec=[rx;ry;rz]; r_norm=norm(r_vec,2);
                     n_new=n_new+1;
                  end
                  fprintf('   Newton method converged in %1i steps\n',n_new);
                  fprintf('   ||r_nl(%2i)||_2=%10.4e\n',n_new,r_norm); 
                  mx=[mx,mx_n]; my=[my,my_n]; mz=[mz,mz_n];
                  if(field==2)
                     me=mx_n*ex+my_n*ey+mz_n*ez;
                     hmcx=[hmcx,k1*me*ex]; hmcy=[hmcy,k1*me*ey]; hmcz=[hmcz,k1*me*ez];
                  end
                  fprintf('   IMR:  mx(%8.4f)=%7.4f; my(%8.4f)=%7.4f; mz(%8.4f)=%7.4f\n',t_c,mx_n,t_c,my_n,t_c,mz_n);
                  fprintf(fid,'   IMR:  mx(%8.4f)=%7.4f; my(%8.4f)=%7.4f; mz(%8.4f)=%7.4f\n',t_c,mx_n,t_c,my_n,t_c,mz_n);
                  dtc=3*dtc; dt=[dt,dtc]; th=[th,t_c]; t_c=t_c+dtc; 
               end
               %  General step size
               fin=0; th=[th,t_c];
               while(t_c<=t_end)
               %  GL predictor step
                  n=n+1;
                  hx_n=0; hy_n=0; hz_n=-1.1; 
                  hx=[hx,hx_n]; hy=[hy,hy_n]; hz=[hz,hz_n];
                  fprintf('STEP %5i, t=%8.3e, dt=%8.3e\n',n,t_c,dt(n));
                  fprintf(fid,'STEP %5i, t=%8.3e, dt=%8.3e\n',n,t_c,dt(n));
                  w=dt(n)/dt(n-1);
                  switch(field)
                     case(1) 
                        mx_gl=(1-w^2)*mx(n)+w^2*mx(n-1)+(1+w)*dt(n)*...
                              (-1/(1+alpha^2)*(my(n)*hz(n)-mz(n)*hy(n))-alpha/(1+alpha^2)*...
                              (my(n)*mx(n)*hy(n)+mx(n)*mz(n)*hz(n)-(my(n)^2+mz(n)^2)*hx(n)));
                        my_gl=(1-w^2)*my(n)+w^2*my(n-1)+(1+w)*dt(n)*...
                              (-1/(1+alpha^2)*(mz(n)*hx(n)-mx(n)*hz(n))-alpha/(1+alpha^2)*...
                              (mx(n)*my(n)*hx(n)+my(n)*mz(n)*hz(n)-(mx(n)^2+mz(n)^2)*hy(n)));
                        mz_gl=(1-w^2)*mz(n)+w^2*mz(n-1)+(1+w)*dt(n)*...
                              (-1/(1+alpha^2)*(mx(n)*hy(n)-my(n)*hx(n))-alpha/(1+alpha^2)*...
                              (mx(n)*mz(n)*hx(n)+my(n)*mz(n)*hy(n)-(mx(n)^2+my(n)^2)*hz(n)));
                     case(2)
                        me_o=mx(n)*ex+my(n)*ey+mz(n)*ez;
                        hhx_o=hx(n)+k1*me_o*ex; hhy_o=hy(n)+k1*me_o*ey; hhz_o=hz(n)+k1*me_o*ez;
                        mx_gl=(1-w^2)*mx(n)+w^2*mx(n-1)+(1+w)*dt(n)*...
                              (-1/(1+alpha^2)*(my(n)*hhz_o-mz(n)*hhy_o)-alpha/(1+alpha^2)*...
                              (my(n)*mx(n)*hhy_o+mx(n)*mz(n)*hhz_o-(my(n)^2+mz(n)^2)*hhx_o));
                        my_gl=(1-w^2)*my(n)+w^2*my(n-1)+(1+w)*dt(n)*...
                              (-1/(1+alpha^2)*(mz(n)*hhx_o-mx(n)*hhz_o)-alpha/(1+alpha^2)*...
                              (mx(n)*my(n)*hhx_o+my(n)*mz(n)*hhz_o-(mx(n)^2+mz(n)^2)*hhy_o));
                        mz_gl=(1-w^2)*mz(n)+w^2*mz(n-1)+(1+w)*dt(n)*...
                              (-1/(1+alpha^2)*(mx(n)*hhy_o-my(n)*hhx_o)-alpha/(1+alpha^2)*...
                              (mx(n)*mz(n)*hhx_o+my(n)*mz(n)*hhy_o-(mx(n)^2+my(n)^2)*hhz_o));
                  end
                  m_gl=[mx_gl;my_gl;mz_gl];      
              %   BDF2 corrector step
                  mx_n=mx(n); my_n=my(n); mz_n=mz(n);   %   Newton init. guess
                  m_n=[mx_n;my_n;mz_n];
                  switch(field)
                     case(1) 
                        rx=mx_n+dtc*(1+w)/(1+2*w)*(1/(1+alpha^2)*(my_n*hz(n+1)-mz_n*hy(n+1))+alpha/(1+alpha^2)*...
                                        (mx_n*my_n*hy(n+1)+mx_n*mz_n*hz(n+1)-(my_n^2+mz_n^2)*hx(n+1)))-...
                                        (1+w)^2/(1+2*w)*mx(n)+w^2/(1+2*w)*mx(n-1);
                        ry=my_n+dtc*(1+w)/(1+2*w)*(1/(1+alpha^2)*(mz_n*hx(n+1)-mx_n*hz(n+1))+alpha/(1+alpha^2)*...
                                        (my_n*mx_n*hx(n+1)+my_n*mz_n*hz(n+1)-(mx_n^2+mz_n^2)*hy(n+1)))-...
                                        (1+w)^2/(1+2*w)*my(n)+w^2/(1+2*w)*my(n-1);
                        rz=mz_n+dtc*(1+w)/(1+2*w)*(1/(1+alpha^2)*(mx_n*hy(n+1)-my_n*hx(n+1))+alpha/(1+alpha^2)*...
                                        (mz_n*mx_n*hx(n+1)+mz_n*my_n*hy(n+1)-(mx_n^2+my_n^2)*hz(n+1)))-...
                                        (1+w)^2/(1+2*w)*mz(n)+w^2/(1+2*w)*mz(n-1);
                      case(2)
                         me_n=mx_n*ex+my_n*ey+mz_n*ez;
                         hhx_n=hx_n+k1*me_n*ex; hhy_n=hy_n+k1*me_n*ey; hhz_n=hz_n+k1*me_n*ez; 
                         rx=mx_n+dtc*(1+w)/(1+2*w)*(1/(1+alpha^2)*(my_n*hhz_n-mz_n*hhy_n)+alpha/(1+alpha^2)*...
                                        (mx_n*my_n*hhy_n+mx_n*mz_n*hhz_n-(my_n^2+mz_n^2)*hhx_n))-...
                                        (1+w)^2/(1+2*w)*mx(n)+w^2/(1+2*w)*mx(n-1);
                         ry=my_n+dtc*(1+w)/(1+2*w)*(1/(1+alpha^2)*(mz_n*hhx_n-mx_n*hhz_n)+alpha/(1+alpha^2)*...
                                        (my_n*mx_n*hhx_n+my_n*mz_n*hhz_n-(mx_n^2+mz_n^2)*hhy_n))-...
                                        (1+w)^2/(1+2*w)*my(n)+w^2/(1+2*w)*my(n-1);
                         rz=mz_n+dtc*(1+w)/(1+2*w)*(1/(1+alpha^2)*(mx_n*hhy_n-my_n*hhx_n)+alpha/(1+alpha^2)*...
                                        (mz_n*mx_n*hhx_n+mz_n*my_n*hhy_n-(mx_n^2+my_n^2)*hhz_n))-...
                                        (1+w)^2/(1+2*w)*mz(n)+w^2/(1+2*w)*mz(n-1);
                  end
                  r_vec=[rx;ry;rz]; r0_norm=norm(r_vec,2); n_new=0; r_norm=1;
                  fprintf('   ||r_nl(0)||_2=%10.4e\n',r0_norm);
                  while(r_norm>=max(1.e-9*r0_norm,1.2e-13))   
                     switch(field)
                        case(1)
                           j11=1+(1+w)*(alpha*dtc)/((1+2*w)*(1+alpha^2))*(my_n*hy(n+1)+mz_n*hz(n+1));
                           j12=((1+w)*dtc)/(1+2*w)*(1/(1+alpha^2)*hz(n+1)+alpha/(1+alpha^2)*...
                                       (mx_n*hy(n+1)-2*my_n*hx(n+1)));
                           j13=((1+w)*dtc)/(1+2*w)*(-1/(1+alpha^2)*hy(n+1)+alpha/(1+alpha^2)*...
                                       (mx_n*hz(n+1)-2*mz_n*hx(n+1)));
                           j21=((1+w)*dtc)/(1+2*w)*(-1/(1+alpha^2)*hz(n+1)+alpha/(1+alpha^2)*...
                                       (my_n*hx(n+1)-2*mx_n*hy(n+1)));
                           j22=1+(1+w)*(alpha*dtc)/((1+2*w)*(1+alpha^2))*(mx_n*hx(n+1)+mz_n*hz(n+1));
                           j23=((1+w)*dtc)/(1+2*w)*(1/(1+alpha^2)*hx(n+1)+alpha/(1+alpha^2)*...
                                       (my_n*hz(n+1)-2*mz_n*hy(n+1)));
                           j31=((1+w)*dtc)/(1+2*w)*(1/(1+alpha^2)*hy(n+1)+alpha/(1+alpha^2)*...
                                       (mz_n*hx(n+1)-2*mx_n*hz(n+1)));
                           j32=((1+w)*dtc)/(1+2*w)*(-1/(1+alpha^2)*hx(n+1)+alpha/(1+alpha^2)*...
                                       (mz_n*hy(n+1)-2*my_n*hz(n+1)));
                           j33=1+(1+w)*(alpha*dtc)/((1+2*w)*(1+alpha^2))*(mx_n*hx(n+1)+my_n*hy(n+1));        
                        case(2)
                           j11=1+((1+w)*dtc)/((1+2*w)*(1+alpha^2))*(k1*(my_n*ex*ez-mz_n*ex*ey))+...
                               ((1+w)*alpha*dtc)/((1+2*w)*(1+alpha^2))*(my_n*hhy_n+k1*my_n*mx_n*ex*ey+...
                                mz_n*hhz_n+k1*mx_n*mz_n*ex*ez-k1*(my_n^2+mz_n^2)*ex*ex); 
                           j12=((1+w)*dtc)/((1+2*w)*(1+alpha^2))*(hhz_n+k1*my_n*ey*ez-k1*mz_n*ey*ey)+...
                               ((1+w)*alpha*dtc)/((1+2*w)*(1+alpha^2))*(mx_n*hhy_n+k1*my_n*mx_n*ey*ey+...
                                k1*mx_n*mz_n*ey*ez-2*my_n*hhx_n-k1*(my_n^2+mz_n^2)*ey*ex);
                           j13=((1+w)*dtc)/((1+2*w)*(1+alpha^2))*(k1*my_n*ez*ez-hhy_n-k1*mz_n*ez*ey)+...
                               ((1+w)*alpha*dtc)/((1+2*w)*(1+alpha^2))*(k1*my_n*mx_n*ez*ey+mx_n*hhz_n+...
                                k1*mx_n*mz_n*ez*ez-2*mz_n*hhx_n-k1*(my_n^2+mz_n^2)*ez*ex);
                           j21=((1+w)*dtc)/((1+2*w)*(1+alpha^2))*(k1*mz_n*ex*ex-hhz_n-k1*mx_n*ex*ez)+...
                               ((1+w)*alpha*dtc)/((1+2*w)*(1+alpha^2))*(my_n*hhx_n+k1*mx_n*my_n*ex*ex+...
                                k1*my_n*mz_n*ex*ez-2*mx_n*hhy_n-k1*(mx_n^2+mz_n^2)*ex*ey);
                           j22=1+((1+w)*dtc)/((1+2*w)*(1+alpha^2))*(k1*(mz_n*ey*ex-mx_n*ey*ez))+...
                               ((1+w)*alpha*dtc)/((1+2*w)*(1+alpha^2))*(mx_n*hhx_n+k1*mx_n*my_n*ey*ex+...
                                mz_n*hhz_n+k1*my_n*mz_n*ey*ez-k1*(mx_n^2+mz_n^2)*ey*ey);
                           j23=((1+w)*dtc)/((1+2*w)*(1+alpha^2))*(hhx_n+k1*mz_n*ez*ex-k1*mx_n*ez*ez)+...
                               ((1+w)*alpha*dtc)/((1+2*w)*(1+alpha^2))*(k1*mx_n*my_n*ez*ex+my_n*hhz_n+...
                                k1*my_n*mz_n*ez*ez-2*mz_n*hhy_n-k1*(mx_n^2+mz_n^2)*ez*ey);
                           j31=((1+w)*dtc)/((1+2*w)*(1+alpha^2))*(hhy_n+k1*mx_n*ex*ey-k1*my_n*ex*ex)+...
                               ((1+w)*alpha*dtc)/((1+2*w)*(1+alpha^2))*(mz_n*hhx_n+k1*mx_n*mz_n*ex*ex+...
                                k1*my_n*mz_n*ex*ey-2*mx_n*hhz_n-k1*(mx_n^2+my_n^2)*ex*ez);
                           j32=((1+w)*dtc)/((1+2*w)*(1+alpha^2))*(k1*mx_n*ey*ey-hhx_n-k1*my_n*ey*ex)+...
                               ((1+w)*alpha*dtc)/((1+2*w)*(1+alpha^2))*(k1*mx_n*mz_n*ey*ex+mz_n*hhy_n+...
                                k1*my_n*mz_n*ey*ey-2*my_n*hhz_n-k1*(mx_n^2+my_n^2)*ey*ez);
                           j33=1+((1+w)*dtc)/((1+2*w)*(1+alpha^2))*(k1*(mx_n*ez*ey-my_n*ez*ex))+...
                              ((1+w)*alpha*dtc)/((1+2*w)*(1+alpha^2))*(mx_n*hhx_n+k1*mx_n*mz_n*ez*ex+...
                               my_n*hhy_n+k1*my_n*mz_n*ez*ey-k1*(mx_n^2+my_n^2)*ez*ez);  
                     end
                     J=[j11,j12,j13;j21,j22,j23;j31,j32,j33];
                     dm_n=J\r_vec;
                     m_n=m_n-dm_n;
                     mx_n=m_n(1); my_n=m_n(2); mz_n=m_n(3);
                     switch(field)
                        case(1) 
                           rx=mx_n+dtc*(1+w)/(1+2*w)*(1/(1+alpha^2)*(my_n*hz(n+1)-mz_n*hy(n+1))+alpha/(1+alpha^2)*...
                                           (mx_n*my_n*hy(n+1)+mx_n*mz_n*hz(n+1)-(my_n^2+mz_n^2)*hx(n+1)))-...
                                           (1+w)^2/(1+2*w)*mx(n)+w^2/(1+2*w)*mx(n-1);
                           ry=my_n+dtc*(1+w)/(1+2*w)*(1/(1+alpha^2)*(mz_n*hx(n+1)-mx_n*hz(n+1))+alpha/(1+alpha^2)*...
                                           (my_n*mx_n*hx(n+1)+my_n*mz_n*hz(n+1)-(mx_n^2+mz_n^2)*hy(n+1)))-...
                                           (1+w)^2/(1+2*w)*my(n)+w^2/(1+2*w)*my(n-1);
                           rz=mz_n+dtc*(1+w)/(1+2*w)*(1/(1+alpha^2)*(mx_n*hy(n+1)-my_n*hx(n+1))+alpha/(1+alpha^2)*...
                                           (mz_n*mx_n*hx(n+1)+mz_n*my_n*hy(n+1)-(mx_n^2+my_n^2)*hz(n+1)))-...
                                           (1+w)^2/(1+2*w)*mz(n)+w^2/(1+2*w)*mz(n-1);
                        case(2)
                           me_n=mx_n*ex+my_n*ey+mz_n*ez;
                           hhx_n=hx_n+k1*me_n*ex; hhy_n=hy_n+k1*me_n*ey; hhz_n=hz_n+k1*me_n*ez; 
                           rx=mx_n+dtc*(1+w)/(1+2*w)*(1/(1+alpha^2)*(my_n*hhz_n-mz_n*hhy_n)+alpha/(1+alpha^2)*...
                                          (mx_n*my_n*hhy_n+mx_n*mz_n*hhz_n-(my_n^2+mz_n^2)*hhx_n))-...
                                          (1+w)^2/(1+2*w)*mx(n)+w^2/(1+2*w)*mx(n-1);
                           ry=my_n+dtc*(1+w)/(1+2*w)*(1/(1+alpha^2)*(mz_n*hhx_n-mx_n*hhz_n)+alpha/(1+alpha^2)*...
                                          (my_n*mx_n*hhx_n+my_n*mz_n*hhz_n-(mx_n^2+mz_n^2)*hhy_n))-...
                                          (1+w)^2/(1+2*w)*my(n)+w^2/(1+2*w)*my(n-1);
                           rz=mz_n+dtc*(1+w)/(1+2*w)*(1/(1+alpha^2)*(mx_n*hhy_n-my_n*hhx_n)+alpha/(1+alpha^2)*...
                                          (mz_n*mx_n*hhx_n+mz_n*my_n*hhy_n-(mx_n^2+my_n^2)*hhz_n))-...
                                          (1+w)^2/(1+2*w)*mz(n)+w^2/(1+2*w)*mz(n-1);
                     end
                     r_vec=[rx;ry;rz]; r_norm=norm(r_vec,2);
                     n_new=n_new+1;
                  end
                  fprintf('   Newton method converged in %1i steps\n',n_new);
                  fprintf('   ||r_nl(%2i)||_2=%10.4e\n',n_new,r_norm); 
                  mx=[mx,mx_n]; my=[my,my_n]; mz=[mz,mz_n];
                  if(field==2)
                     me=mx_n*ex+my_n*ey+mz_n*ez;
                     hmcx=[hmcx,k1*me*ex]; hmcy=[hmcy,k1*me*ey]; hmcz=[hmcz,k1*me*ez];
                  end
                  fprintf('   GL:    mx(%8.4f)=%7.4f; my(%8.4f)=%7.4f; mz(%8.4f)=%7.4f\n',t_c,mx_gl,t_c,my_gl,t_c,mz_gl);
                  fprintf(fid,'   GL:    mx(%8.4f)=%7.4f; my(%8.4f)=%7.4f; mz(%8.4f)=%7.4f\n',t_c,mx_gl,t_c,my_gl,t_c,mz_gl);
                  fprintf('   BDF2:  mx(%8.4f)=%7.4f; my(%8.4f)=%7.4f; mz(%8.4f)=%7.4f\n',t_c,mx_n,t_c,my_n,t_c,mz_n);
                  fprintf(fid,'   BDF2:  mx(%8.4f)=%7.4f; my(%8.4f)=%7.4f; mz(%8.4f)=%7.4f\n',t_c,mx_n,t_c,my_n,t_c,mz_n);
                  %   LTE estimate
                  if(fin==1)
                     break;
                  end
                  d=(1+1/w)^2/(1+3/w+4/(w^2)+2/(w^3))*(m_n-m_gl); dh=[dh,norm(d,2)];
                  fprintf('   ||d(%5i)||=%8.3e\n',n,norm(d,2));
                  fprintf(fid,'   ||d(%5i)||=%8.3e\n',n,norm(d,2));
                  dt_new=(eps_t/norm(d,2))^(1/3)*dt(n); 
                  if(t_c+dt_new>=t_end)
                     dt_new=abs(t_end-t_c);
                     t_c=t_end; fin=1;
                  else
                     t_c=t_c+dt_new;
                  end
                  dt=[dt,dt_new]; th=[th,t_c]; dtc=dt_new; 
               end
               set(gcf,'Position',[575,240,700,700]);
               subplot(221); plot(th,mx,'r.-','MarkerSize',4); 
               axis([0 t_end -1.01 1.01]);
               title('x-component of the magnetization m');
               xlabel('t'); ylabel('mx');
               subplot(222); plot(th,my,'b.-','MarkerSize',4);
               axis([0 t_end -1.01 1.01]);
               title('y-component of the magnetization m');
               xlabel('t'); ylabel('my');
               subplot(223); plot(th,mz,'g.-','MarkerSize',4);
               axis([0 t_end -1.01 1.01]);
               title('z-component of the magnetization m');
               xlabel('t'); ylabel('mz');
               subplot(224); plot(th,[0,dt],'k.-','MarkerSize',4);
               axis([0 t_end 0 1.1*max(dt)]);
               title('Time step history');
               xlabel('t'); ylabel('dt');
            otherwise
              fprintf('Wrong integration mode\n');  
       end 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                  AB2-TR                                 %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%       
    case(2)   %  AB2-TR
       switch(mode)
            case(1)   %   fixed step size
               t_c=dtc;   %   current time
               n=0;       %   step counter
               dt=[dtc];  %   time step history
               th=[0];    %   discrete time history
               fin=0;     %   finished flag
               while(t_c<=t_end && fin==0)
                  n=n+1;
                  fprintf('STEP %5i, t=%8.3e, dt=%8.3e\n',n,t_c,dtc);
                  fprintf(fid,'STEP %5i, t=%8.3e, dt=%8.3e\n',n,t_c,dtc);
                  mx_n=mx(n); my_n=my(n); mz_n=mz(n);   %   Newton init. guess
                  m_n=[mx_n;my_n;mz_n];
                  hx_n=0; hy_n=0; hz_n=-1.1;   %  external field
                  hx=[hx,hx_n]; hy=[hy,hy_n]; hz=[hz,hz_n];
                  switch(field)
                     case(1)
                        rx=mx_n+dtc/2*(1/(1+alpha^2)*(my_n*hz(n+1)-mz_n*hy(n+1))+alpha/(1+alpha^2)*...
                                (mx_n*my_n*hy(n+1)+mx_n*mz_n*hz(n+1)-(my_n^2+mz_n^2)*hx(n)))-mx(n)+...
                                +dtc/2*(1/(1+alpha^2)*(my(n)*hz(n)-mz(n)*hy(n))+alpha/(1+alpha^2)*...
                                (mx(n)*my(n)*hy(n)+mx(n)*mz(n)*hz(n)-(my(n)^2+mz(n)^2)*hx(n)));
                        ry=my_n+dtc/2*(1/(1+alpha^2)*(mz_n*hx(n+1)-mx_n*hz(n+1))+alpha/(1+alpha^2)*...
                                (my_n*mx_n*hx(n+1)+my_n*mz_n*hz(n+1)-(mx_n^2+mz_n^2)*hy(n+1)))-my(n)+...
                                +dtc/2*(1/(1+alpha^2)*(mz(n)*hx(n)-mx(n)*hz(n))+alpha/(1+alpha^2)*...
                                (my(n)*mx(n)*hx(n)+my(n)*mz(n)*hz(n)-(mx(n)^2+mz(n)^2)*hy(n)));
                        rz=mz_n+dtc/2*(1/(1+alpha^2)*(mx_n*hy(n+1)-my_n*hx(n+1))+alpha/(1+alpha^2)*...
                                (mz_n*mx_n*hx(n+1)+mz_n*my_n*hy(n+1)-(mx_n^2+my_n^2)*hz(n+1)))-mz(n)+...
                                +dtc/2*(1/(1+alpha^2)*(mx(n)*hy(n)-my(n)*hx(n))+alpha/(1+alpha^2)*...
                                (mz(n)*mx(n)*hx(n)+mz(n)*my(n)*hy(n)-(mx(n)^2+my(n)^2)*hz(n))); 
                     case(2)
                        me_n=mx_n*ex+my_n*ey+mz_n*ez;
                        hhx_n=hx_n+k1*me_n*ex; hhy_n=hy_n+k1*me_n*ey; hhz_n=hz_n+k1*me_n*ez;
                        me_o=mx(n)*ex+my(n)*ey+mz(n)*ez;
                        hhx_o=hx(n)+k1*me_o*ex; hhy_o=hy(n)+k1*me_o*ey; hhz_o=hz(n)+k1*me_o*ez;
                        rx=mx_n+dtc/2*(1/(1+alpha^2)*(my_n*hhz_n-mz_n*hhy_n)+alpha/(1+alpha^2)*...
                                (mx_n*my_n*hhy_n+mx_n*mz_n*hhz_n-(my_n^2+mz_n^2)*hhx_n))-mx(n)+...
                                +dtc/2*(1/(1+alpha^2)*(my(n)*hhz_o-mz(n)*hhy_o)+alpha/(1+alpha^2)*...
                                (mx(n)*my(n)*hhy_o+mx(n)*mz(n)*hhz_o-(my(n)^2+mz(n)^2)*hhx_o));
                        ry=my_n+dtc/2*(1/(1+alpha^2)*(mz_n*hhx_n-mx_n*hhz_n)+alpha/(1+alpha^2)*...
                                (my_n*mx_n*hhx_n+my_n*mz_n*hhz_n-(mx_n^2+mz_n^2)*hhy_n))-my(n)+...
                                +dtc/2*(1/(1+alpha^2)*(mz(n)*hhx_o-mx(n)*hhz_o)+alpha/(1+alpha^2)*...
                                (my(n)*mx(n)*hhx_o+my(n)*mz(n)*hhz_o-(mx(n)^2+mz(n)^2)*hhy_o));
                        rz=mz_n+dtc/2*(1/(1+alpha^2)*(mx_n*hhy_n-my_n*hhx_n)+alpha/(1+alpha^2)*...
                                (mz_n*mx_n*hhx_n+mz_n*my_n*hhy_n-(mx_n^2+my_n^2)*hhz_n))-mz(n)+...
                                +dtc/2*(1/(1+alpha^2)*(mx(n)*hhy_o-my(n)*hhx_o)+alpha/(1+alpha^2)*...
                                (mz(n)*mx(n)*hhx_o+mz(n)*my(n)*hhy_o-(mx(n)^2+my(n)^2)*hhz_o));
                  end
                  r_vec=[rx;ry;rz]; r0_norm=norm(r_vec,2); n_new=0; r_norm=1;
                  fprintf('   ||r_nl(0)||_2=%10.4e\n',r0_norm);
                  while(r_norm>=max(1.e-9*r0_norm,1.2e-13))
                     switch(field)
                        case(1) 
                           j11=1+(alpha*dtc)/(2*(1+alpha^2))*(my_n*hy(n+1)+mz_n*hz(n+1));
                           j12=dtc/(2*(1+alpha^2))*hz(n+1)+(alpha*dtc)/(2*(1+alpha^2))*(mx_n*hy(n+1)-2*my_n*hx(n+1));
                           j13=-dtc/(2*(1+alpha^2))*hy(n+1)+(alpha*dtc)/(2*(1+alpha^2))*(mx_n*hz(n+1)-2*mz_n*hx(n+1));
                           j21=-dtc/(2*(1+alpha^2))*hz(n+1)+(alpha*dtc)/(2*(1+alpha^2))*(my_n*hx(n+1)-2*mx_n*hy(n+1));
                           j22=1+(alpha*dtc)/(2*(1+alpha^2))*(mx_n*hx(n+1)+mz_n*hz(n+1));
                           j23=dtc/(2*(1+alpha^2))*hx(n+1)+(alpha*dtc)/(2*(1+alpha^2))*(my_n*hz(n+1)-2*mz_n*hy(n+1));
                           j31=dtc/(2*(1+alpha^2))*hy(n+1)+(alpha*dtc)/(2*(1+alpha^2))*(mz_n*hx(n+1)-2*mx_n*hz(n+1));
                           j32=-dtc/(2*(1+alpha^2))*hx(n+1)+(alpha*dtc)/(2*(1+alpha^2))*(mz_n*hy(n+1)-2*my_n*hz(n+1));
                           j33=1+(alpha*dtc)/(2*(1+alpha^2))*(mx_n*hx(n+1)+my_n*hy(n+1));
                        case(2)
                           j11=1+dtc/(2*(1+alpha^2))*(k1*(my_n*ex*ez-mz_n*ex*ey))+...
                                 (alpha*dtc)/(2*(1+alpha^2))*(my_n*hhy_n+k1*my_n*mx_n*ex*ey+mz_n*hhz_n+k1*mx_n*mz_n*ex*ez-...
                                  k1*(my_n^2+mz_n^2)*ex*ex);
                           j12=dtc/(2*(1+alpha^2))*(hhz_n+k1*my_n*ey*ez-k1*mz_n*ey*ey)+...
                               (alpha*dtc)/(2*(1+alpha^2))*(mx_n*hhy_n+k1*my_n*mx_n*ey*ey+k1*mx_n*mz_n*ey*ez-...
                                2*my_n*hhx_n-k1*(my_n^2+mz_n^2)*ey*ex);
                           j13=dtc/(2*(1+alpha^2))*(k1*my_n*ez*ez-hhy_n-k1*mz_n*ez*ey)+...
                               (alpha*dtc)/(2*(1+alpha^2))*(k1*my_n*mx_n*ez*ey+mx_n*hhz_n+k1*mx_n*mz_n*ez*ez-...
                                2*mz_n*hhx_n-k1*(my_n^2+mz_n^2)*ez*ex);
                           j21=dtc/(2*(1+alpha^2))*(k1*mz_n*ex*ex-hhz_n-k1*mx_n*ex*ez)+...
                               (alpha*dtc)/(2*(1+alpha^2))*(my_n*hhx_n+k1*mx_n*my_n*ex*ex+...
                                k1*my_n*mz_n*ex*ez-2*mx_n*hhy_n-k1*(mx_n^2+mz_n^2)*ex*ey);
                           j22=1+dtc/(2*(1+alpha^2))*(k1*(mz_n*ey*ex-mx_n*ey*ez))+...
                                 (alpha*dtc)/(2*(1+alpha^2))*(mx_n*hhx_n+k1*mx_n*my_n*ey*ex+mz_n*hhz_n+k1*my_n*mz_n*ey*ez-...
                                  k1*(mx_n^2+mz_n^2)*ey*ey);
                           j23=dtc/(2*(1+alpha^2))*(hhx_n+k1*mz_n*ez*ex-k1*mx_n*ez*ez)+...
                               (alpha*dtc)/(2*(1+alpha^2))*(k1*mx_n*my_n*ez*ex+my_n*hhz_n+k1*my_n*mz_n*ez*ez-...
                                2*mz_n*hhy_n-k1*(mx_n^2+mz_n^2)*ez*ey);
                           j31=dtc/(2*(1+alpha^2))*(hhy_n+k1*mx_n*ex*ey-k1*my_n*ex*ex)+...
                               (alpha*dtc)/(2*(1+alpha^2))*(mz_n*hhx_n+k1*mx_n*mz_n*ex*ex+k1*my_n*mz_n*ex*ey-...
                                2*mx_n*hhz_n-k1*(mx_n^2+my_n^2)*ex*ez);
                           j32=dtc/(2*(1+alpha^2))*(k1*mx_n*ey*ey-hhx_n-k1*my_n*ey*ex)+...
                               (alpha*dtc)/(2*(1+alpha^2))*(k1*mx_n*mz_n*ey*ex+mz_n*hhy_n+k1*my_n*mz_n*ey*ey-...
                                2*my_n*hhz_n-k1*(mx_n^2+my_n^2)*ey*ez);
                           j33=1+dtc/(2*(1+alpha^2))*(k1*(mx_n*ez*ey-my_n*ez*ex))+...
                                 (alpha*dtc)/(2*(1+alpha^2))*(mx_n*hhx_n+k1*mx_n*mz_n*ez*ex+my_n*hhy_n+k1*my_n*mz_n*ez*ey-...
                                  k1*(mx_n^2+my_n^2)*ez*ez);  
                     end
                     J=[j11,j12,j13;j21,j22,j23;j31,j32,j33];
                     dm_n=J\r_vec;
                     m_n=m_n-dm_n;
                     mx_n=m_n(1); my_n=m_n(2); mz_n=m_n(3);
                     switch(field)
                        case(1)
                           rx=mx_n+dtc/2*(1/(1+alpha^2)*(my_n*hz(n+1)-mz_n*hy(n+1))+alpha/(1+alpha^2)*...
                                   (mx_n*my_n*hy(n+1)+mx_n*mz_n*hz(n+1)-(my_n^2+mz_n^2)*hx(n+1)))-mx(n)+...
                                   +dtc/2*(1/(1+alpha^2)*(my(n)*hz(n)-mz(n)*hy(n))+alpha/(1+alpha^2)*...
                                   (mx(n)*my(n)*hy(n)+mx(n)*mz(n)*hz(n)-(my(n)^2+mz(n)^2)*hx(n)));
                           ry=my_n+dtc/2*(1/(1+alpha^2)*(mz_n*hx(n+1)-mx_n*hz(n+1))+alpha/(1+alpha^2)*...
                                   (my_n*mx_n*hx(n+1)+my_n*mz_n*hz(n+1)-(mx_n^2+mz_n^2)*hy(n+1)))-my(n)+...
                                   +dtc/2*(1/(1+alpha^2)*(mz(n)*hx(n)-mx(n)*hz(n))+alpha/(1+alpha^2)*...
                                   (my(n)*mx(n)*hx(n)+my(n)*mz(n)*hz(n)-(mx(n)^2+mz(n)^2)*hy(n)));
                           rz=mz_n+dtc/2*(1/(1+alpha^2)*(mx_n*hy(n+1)-my_n*hx(n+1))+alpha/(1+alpha^2)*...
                                   (mz_n*mx_n*hx(n+1)+mz_n*my_n*hy(n+1)-(mx_n^2+my_n^2)*hz(n+1)))-mz(n)+...
                                   +dtc/2*(1/(1+alpha^2)*(mx(n)*hy(n)-my(n)*hx(n))+alpha/(1+alpha^2)*...
                                   (mz(n)*mx(n)*hx(n)+mz(n)*my(n)*hy(n)-(mx(n)^2+my(n)^2)*hz(n))); 
                        case(2)
                           me_n=mx_n*ex+my_n*ey+mz_n*ez;
                           hhx_n=hx_n+k1*me_n*ex; hhy_n=hy_n+k1*me_n*ey; hhz_n=hz_n+k1*me_n*ez;
                           rx=mx_n+dtc/2*(1/(1+alpha^2)*(my_n*hhz_n-mz_n*hhy_n)+alpha/(1+alpha^2)*...
                                   (mx_n*my_n*hhy_n+mx_n*mz_n*hhz_n-(my_n^2+mz_n^2)*hhx_n))-mx(n)+...
                                   +dtc/2*(1/(1+alpha^2)*(my(n)*hhz_o-mz(n)*hhy_o)+alpha/(1+alpha^2)*...
                                   (mx(n)*my(n)*hhy_o+mx(n)*mz(n)*hhz_o-(my(n)^2+mz(n)^2)*hhx_o));
                           ry=my_n+dtc/2*(1/(1+alpha^2)*(mz_n*hhx_n-mx_n*hhz_n)+alpha/(1+alpha^2)*...
                                   (my_n*mx_n*hhx_n+my_n*mz_n*hhz_n-(mx_n^2+mz_n^2)*hhy_n))-my(n)+...
                                   +dtc/2*(1/(1+alpha^2)*(mz(n)*hhx_o-mx(n)*hhz_o)+alpha/(1+alpha^2)*...
                                   (my(n)*mx(n)*hhx_o+my(n)*mz(n)*hhz_o-(mx(n)^2+mz(n)^2)*hhy_o));
                           rz=mz_n+dtc/2*(1/(1+alpha^2)*(mx_n*hhy_n-my_n*hhx_n)+alpha/(1+alpha^2)*...
                                   (mz_n*mx_n*hhx_n+mz_n*my_n*hhy_n-(mx_n^2+my_n^2)*hhz_n))-mz(n)+...
                                   +dtc/2*(1/(1+alpha^2)*(mx(n)*hhy_o-my(n)*hhx_o)+alpha/(1+alpha^2)*...
                                   (mz(n)*mx(n)*hhx_o+mz(n)*my(n)*hhy_o-(mx(n)^2+my(n)^2)*hhz_o)); 
                     end
                     r_vec=[rx;ry;rz]; r_norm=norm(r_vec,2);
                     n_new=n_new+1;
                  end
                  fprintf('   Newton method converged in %1i steps\n',n_new);
                  fprintf('   ||r_nl(%2i)||_2=%10.4e\n',n_new,r_norm); 
                  mx=[mx,mx_n]; my=[my,my_n]; mz=[mz,mz_n];
                  if(field==2)
                     me=mx_n*ex+my_n*ey+mz_n*ez;
                     hmcx=[hmcx,k1*me*ex]; hmcy=[hmcy,k1*me*ey]; hmcz=[hmcz,k1*me*ez];
                  end
                  fprintf('   TR:  mx(%8.4f)=%7.4f; my(%8.4f)=%7.4f; mz(%8.4f)=%7.4f\n',t_c,mx_n,t_c,my_n,t_c,mz_n);
                  fprintf(fid,'   TR:  mx(%8.4f)=%7.4f; my(%8.4f)=%7.4f; mz(%8.4f)=%7.4f\n',t_c,mx_n,t_c,my_n,t_c,mz_n);
                  if(t_c==t_end)
                     fin=1;
                  end
                  t_c=t_c+dtc;
                  if(t_c>=t_end)
                     dtc=dtc-t_c+t_end; 
                     t_c=t_end; 
                     if(dtc<=1.e-6)
                        fin=1;
                     end
                  end
                  dt=[dt,dtc]; th=[th,t_c]; 
               end
               set(gcf,'Position',[575,240,700,700]);
               subplot(221); plot(th,mx,'r.-','MarkerSize',4); 
               axis([0 t_end -1.01 1.01]);
               title('x-component of the magnetization m');
               xlabel('t'); ylabel('mx');
               subplot(222); plot(th,my,'b.-','MarkerSize',4);
               axis([0 t_end -1.01 1.01]);
               title('y-component of the magnetization m');
               xlabel('t'); ylabel('my');
               subplot(223); plot(th,mz,'g.-','MarkerSize',4);
               axis([0 t_end -1.01 1.01]);
               title('z-component of the magnetization m');
               xlabel('t'); ylabel('mz');
               subplot(224); plot(th,dt,'k.-','MarkerSize',4);
               axis([0 t_end 0 1.1*max(dt)]);
               title('Time step history');
               xlabel('t'); ylabel('dt');
            case(2)   %   variable step size %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
               dt=[dt0]; %  the time step history
               t_c=dt0;  %  current time
               dtc=dt0;  %  current step size
               th=[0];   %  time history
               dh=[0,0,0]; %  LTE estimate history
               dexh=[0,0,0];  %  exact LTE history
               %  Startup (1 steps of IMR with dt(0))
               for n=1:1 
                  fprintf('STEP %5i, t=%8.3e\n',n,t_c);
                  fprintf(fid,'STEP %5i, t=%8.3e\n',n,t_c);
                  mx_n=mx(n); my_n=my(n); mz_n=mz(n);   %   Newton init. guess
                  m_n=[mx_n;my_n;mz_n];
                  hx_n=0; hy_n=0; hz_n=-1.1;   %  external field
                  hx=[hx,hx_n]; hy=[hy,hy_n]; hz=[hz,hz_n];
                  mx_h=(mx_n+mx(n))/2; my_h=(my_n+my(n))/2; mz_h=(mz_n+mz(n))/2;
                  hx_h=(hx(n+1)+hx(n))/2; hy_h=(hy(n+1)+hy(n))/2; hz_h=(hz(n+1)+hz(n))/2;
                  switch(field)
                     case(1)
                        rx=mx_n+dtc*(1/(1+alpha^2)*(my_h*hz_h-mz_h*hy_h)+alpha/(1+alpha^2)*...
                                    (my_h*mx_h*hy_h+mx_h*mz_h*hz_h-(my_h^2+mz_h^2)*hx_h))-mx(n);
                        ry=my_n+dtc*(1/(1+alpha^2)*(mz_h*hx_h-mx_h*hz_h)+alpha/(1+alpha^2)*...
                                    (mx_h*my_h*hx_h+my_h*mz_h*hz_h-(mx_h^2+mz_h^2)*hy_h))-my(n);
                        rz=mz_n+dtc*(1/(1+alpha^2)*(mx_h*hy_h-my_h*hx_h)+alpha/(1+alpha^2)*...
                                    (mx_h*mz_h*hx_h+my_h*mz_h*hy_h-(mx_h^2+my_h^2)*hz_h))-mz(n); 
                     case(2)
                        me_h=(mx_n+mx(n))/2*ex+(my_n+my(n))/2*ey+(mz_n+mz(n))/2*ez;
                        hhx_h=hx_h+k1*me_h*ex; hhy_h=hy_h+k1*me_h*ey; hhz_h=hz_h+k1*me_h*ez;
                        rx=mx_n+dtc*(1/(1+alpha^2)*(my_h*hhz_h-mz_h*hhy_h)+alpha/(1+alpha^2)*...
                                    (my_h*mx_h*hhy_h+mx_h*mz_h*hhz_h-(my_h^2+mz_h^2)*hhx_h))-mx(n);
                        ry=my_n+dtc*(1/(1+alpha^2)*(mz_h*hhx_h-mx_h*hhz_h)+alpha/(1+alpha^2)*...
                                    (mx_h*my_h*hhx_h+my_h*mz_h*hhz_h-(mx_h^2+mz_h^2)*hhy_h))-my(n);
                        rz=mz_n+dtc*(1/(1+alpha^2)*(mx_h*hhy_h-my_h*hhx_h)+alpha/(1+alpha^2)*...
                                    (mx_h*mz_h*hhx_h+my_h*mz_h*hhy_h-(mx_h^2+my_h^2)*hhz_h))-mz(n);  
                  end
                  r_vec=[rx;ry;rz]; r0_norm=norm(r_vec,2); n_new=0; r_norm=1;
                  fprintf('   ||r_nl(0)||_2=%10.4e\n',r0_norm);
                  while(r_norm>=max(1.e-9*r0_norm,1.2e-15))
                     switch(field)
                        case(1)
                           j11=1+(alpha*dtc)/(2*(1+alpha^2))*(my_h*hy_h+mz_h*hz_h);
                           j12=dtc/(2*(1+alpha^2))*hz_h+(alpha*dtc)/(2*(1+alpha^2))*(mx_h*hy_h-2*my_h*hx_h);
                           j13=-dtc/(2*(1+alpha^2))*hy_h+(alpha*dtc)/(2*(1+alpha^2))*(mx_h*hz_h-2*mz_h*hx_h);
                           j21=-dtc/(2*(1+alpha^2))*hz_h+(alpha*dtc)/(2*(1+alpha^2))*(my_h*hx_h-2*mx_h*hy_h);
                           j22=1+(alpha*dtc)/(2*(1+alpha^2))*(mx_h*hx_h+mz_h*hz_h);
                           j23=dtc/(2*(1+alpha^2))*hx_h+(alpha*dtc)/(2*(1+alpha^2))*(my_h*hz_h-2*mz_h*hy_h);
                           j31=dtc/(2*(1+alpha^2))*hy_h+(alpha*dtc)/(2*(1+alpha^2))*(mz_h*hx_h-2*mx_h*hz_h);
                           j32=-dtc/(2*(1+alpha^2))*hx_h+(alpha*dtc)/(2*(1+alpha^2))*(mz_h*hy_h-2*my_h*hz_h);
                           j33=1+(alpha*dtc)/(2*(1+alpha^2))*(mx_h*hx_h+my_h*hy_h);
                        case(2)
                           j11=1+(dtc/(2*(1+alpha^2)))*(k1*(my_h*ex*ez-mz_h*ex*ey))+(dtc*alpha)/(2*(1+alpha^2))*...
                                 (my_h*hhy_h+k1*my_h*mx_h*ex*ey+mz_h*hhz_h+k1*mx_h*mz_h*ex*ez-k1*(my_h^2+mz_h^2)*ex*ex);
                           j12=dtc/(2*(1+alpha^2))*(hhz_h-k1*mz_h*ey*ey+k1*my_h*ey*ez)+(dtc*alpha)/(2*(1+alpha^2))*...
                                 (mx_h*hy_h+k1*mx_h*my_h*ey*ey+k1*mx_h*mz_h*ey*ez-2*my_h*hhx_h-k1*(my_h^2+mz_h^2)*ey*ex);
                           j13=dtc/(2*(1+alpha^2))*(k1*my_h*ez*ez-hhy_h-k1*mz_h*ez*ey)+(dtc*alpha)/(2*(1+alpha^2))*...
                                 (k1*my_h*mx_h*ez*ey+mx_h*hhz_h+k1*mx_h*mz_h*ez*ez-2*mz_h*hhx_h-k1*(my_h^2+mz_h^2)*ez*ex);
                           j21=dtc/(2*(1+alpha^2))*(k1*mz_h*ex*ex-hhz_h-k1*mx_h*ex*ez)+(dtc*alpha)/(2*(1+alpha^2))*...
                                 (my_h*hhx_h+k1*mx_h*my_h*ex*ex+k1*my_h*mz_h*ex*ez-2*mx_h*hhy_h-k1*(mx_h^2+mz_h^2)*ex*ey);
                           j22=1+(dtc/(2*(1+alpha^2)))*(k1*(mz_h*ey*ex-mx_h*ey*ez))+(dtc*alpha)/(2*(1+alpha^2))*...
                                 (mx_h*hx_h+k1*mx_h*my_h*ey*ex+mz_h*hhz_h+k1*my_h*mz_h*ey*ez-k1*(mx_h^2+mz_h^2)*ey*ey);
                           j23=dtc/(2*(1+alpha^2))*(hhx_h+k1*mz_h*ez*ex-k1*mx_h*ez*ez)+(dtc*alpha)/(2*(1+alpha^2))*...
                                 (k1*mx_h*my_h*ez*ex+my_h*hhz_h+k1*my_h*mz_h*ez*ez-2*mz_h*hhy_h-k1*(mx_h^2+mz_h^2)*ez*ey);
                           j31=dtc/(2*(1+alpha^2))*(hhy_h-k1*my_h*ex*ex+k1*mx*ex*ey)+(dtc*alpha)/(2*(1+alpha^2))*...
                                 (mz_h*hhx_h+k1*mx_h*mz_h*ex*ex+k1*my_h*mz_h*ex*ey-2*mx_h*hhz_h-k1*(mx_h^2+my_h^2)*ex*ez);
                           j32=dtc/(2*(1+alpha^2))*(k1*mx_h*ey*ey-hhx_h-k1*my_h*ey*ex)+(dtc*alpha)/(2*(1+alpha^2))*...
                                 (k1*mx_h*mz_h*ey*ex+mz_h*hhy_h+k1*my_h*mz_h*ey*ey-2*my_h*hhz_h-k1*(mx_h^2+my_h^2)*ey*ez);
                           j33=1+(dtc/(2*(1+alpha^2)))*(k1*(mx_h*ez*ey-my_h*ez*ex))+(dtc*alpha)/(2*(1+alpha^2))*...
                                 (mx_h*hhx_h+k1*mx_h*mz_h*ez*ex+my_h*hhy_h+k1*my_h*mz_h*ez*ey-k1*(mx_h^2+my_h)^2*ez*ez); 
                     end
                     J=[j11,j12,j13;j21,j22,j23;j31,j32,j33];
                     dm_n=J\r_vec;
                     m_n=m_n-dm_n;
                     mx_n=m_n(1); my_n=m_n(2); mz_n=m_n(3);
                     mx_h=(mx_n+mx(n))/2; my_h=(my_n+my(n))/2; mz_h=(mz_n+mz(n))/2;
                     switch(field)
                        case(1)
                           rx=mx_n+dtc*(1/(1+alpha^2)*(my_h*hz_h-mz_h*hy_h)+alpha/(1+alpha^2)*...
                                       (my_h*mx_h*hy_h+mx_h*mz_h*hz_h-(my_h^2+mz_h^2)*hx_h))-mx(n);
                           ry=my_n+dtc*(1/(1+alpha^2)*(mz_h*hx_h-mx_h*hz_h)+alpha/(1+alpha^2)*...
                                       (mx_h*my_h*hx_h+my_h*mz_h*hz_h-(mx_h^2+mz_h^2)*hy_h))-my(n);
                           rz=mz_n+dtc*(1/(1+alpha^2)*(mx_h*hy_h-my_h*hx_h)+alpha/(1+alpha^2)*...
                                       (mx_h*mz_h*hx_h+my_h*mz_h*hy_h-(mx_h^2+my_h^2)*hz_h))-mz(n);
                        case(2)
                           me_h=(mx_n+mx(n))/2*ex+(my_n+my(n))/2*ey+(mz_n+mz(n))/2*ez;
                           hhx_h=hx_h+k1*me_h*ex; hhy_h=hy_h+k1*me_h*ey; hhz_h=hz_h+k1*me_h*ez;
                           rx=mx_n+dtc*(1/(1+alpha^2)*(my_h*hhz_h-mz_h*hhy_h)+alpha/(1+alpha^2)*...
                                       (my_h*mx_h*hhy_h+mx_h*mz_h*hhz_h-(my_h^2+mz_h^2)*hhx_h))-mx(n);
                           ry=my_n+dtc*(1/(1+alpha^2)*(mz_h*hhx_h-mx_h*hhz_h)+alpha/(1+alpha^2)*...
                                       (mx_h*my_h*hhx_h+my_h*mz_h*hhz_h-(mx_h^2+mz_h^2)*hhy_h))-my(n);
                           rz=mz_n+dtc*(1/(1+alpha^2)*(mx_h*hhy_h-my_h*hhx_h)+alpha/(1+alpha^2)*...
                                       (mx_h*mz_h*hhx_h+my_h*mz_h*hhy_h-(mx_h^2+my_h^2)*hhz_h))-mz(n);     
                     end
                     r_vec=[rx;ry;rz]; r_norm=norm(r_vec,2);
                     n_new=n_new+1;
                  end
                  fprintf('   Newton method converged in %1i steps\n',n_new);
                  fprintf('   ||r_nl(%2i)||_2=%10.4e\n',n_new,r_norm); 
                  mx=[mx,mx_n]; my=[my,my_n]; mz=[mz,mz_n];
                  if(field==2)
                     me=mx_n*ex+my_n*ey+mz_n*ez;
                     hmcx=[hmcx,k1*me*ex]; hmcy=[hmcy,k1*me*ey]; hmcz=[hmcz,k1*me*ez];
                  end
                  fprintf('   IMR:  mx(%8.4f)=%7.4f; my(%8.4f)=%7.4f; mz(%8.4f)=%7.4f\n',t_c,mx_n,t_c,my_n,t_c,mz_n);
                  fprintf(fid,'   IMR:  mx(%8.4f)=%7.4f; my(%8.4f)=%7.4f; mz(%8.4f)=%7.4f\n',t_c,mx_n,t_c,my_n,t_c,mz_n);
                  dtc=3*dtc; dt=[dt,dtc]; th=[th,t_c]; t_c=t_c+dtc; 
               end
               %  General step size
               fin=0; th=[th,t_c];
               while(t_c<=t_end)
               %  AB2 predictor step
                  n=n+1;
                  fprintf('STEP %5i, t=%8.3e, dt=%8.3e\n',n,t_c,dt(n));
                  fprintf(fid,'STEP %5i, t=%8.3e, dt=%8.3e\n',n,t_c,dt(n));
                  switch(field)
                     case(1) 
                        mx_ab2=mx(n)+(dt(n)+dt(n)^2/(2*dt(n-1)))*(-1/(1+alpha^2)*(my(n)*hz(n)-mz(n)*hy(n))-...
                                      alpha/(1+alpha^2)*(my(n)*mx(n)*hy(n)+mx(n)*mz(n)*hz(n)-(my(n)^2+mz(n)^2)*hx(n)))-...
                                      dt(n)^2/(2*dt(n-1))*(-1/(1+alpha^2)*(my(n-1)*hz(n-1)-mz(n-1)*hy(n-1))-...
                                      alpha/(1+alpha^2)*(my(n-1)*mx(n-1)*hy(n-1)+mx(n-1)*mz(n-1)*hz(n-1)-...
                                      (my(n-1)^2+mz(n-1)^2)*hx(n-1)));
                        my_ab2=my(n)+(dt(n)+dt(n)^2/(2*dt(n-1)))*(-1/(1+alpha^2)*(mz(n)*hx(n)-mx(n)*hz(n))-...
                                      alpha/(1+alpha^2)*(mx(n)*my(n)*hx(n)+my(n)*mz(n)*hz(n)-(mx(n)^2+mz(n)^2)*hy(n)))-...
                                      dt(n)^2/(2*dt(n-1))*(-1/(1+alpha^2)*(mz(n-1)*hx(n-1)-mx(n-1)*hz(n-1))-...
                                      alpha/(1+alpha^2)*(mx(n-1)*my(n-1)*hx(n-1)+my(n-1)*mz(n-1)*hz(n-1)-...
                                      (mx(n-1)^2+mz(n-1)^2)*hy(n-1)));
                        mz_ab2=mz(n)+(dt(n)+dt(n)^2/(2*dt(n-1)))*(-1/(1+alpha^2)*(mx(n)*hy(n)-my(n)*hx(n))-...
                                      alpha/(1+alpha^2)*(mx(n)*mz(n)*hx(n)+my(n)*mz(n)*hy(n)-(mx(n)^2+my(n)^2)*hz(n)))-...
                                      dt(n)^2/(2*dt(n-1))*(-1/(1+alpha^2)*(mx(n-1)*hy(n-1)-my(n-1)*hx(n-1))-...
                                      alpha/(1+alpha^2)*(mx(n-1)*mz(n-1)*hx(n-1)+my(n-1)*mz(n-1)*hy(n-1)-...
                                      (mx(n-1)^2+my(n-1)^2)*hz(n-1)));
                     case(2)  
                        me_o1=mx(n)*ex+my(n)*ey+mz(n)*ez;
                        hhx_o1=hx(n)+k1*me_o1*ex; hhy_o1=hy(n)+k1*me_o1*ey; hhz_o1=hz(n)+k1*me_o1*ez;
                        me_o2=mx(n-1)*ex+my(n-1)*ey+mz(n-1)*ez;
                        hhx_o2=hx(n-1)+k1*me_o2*ex; hhy_o2=hy(n-1)+k1*me_o2*ey; hhz_o2=hz(n-1)+k1*me_o2*ez;
                        mx_ab2=mx(n)+(dt(n)+dt(n)^2/(2*dt(n-1)))*(-1/(1+alpha^2)*(my(n)*hhz_o1-mz(n)*hhy_o1)-...
                                      alpha/(1+alpha^2)*(my(n)*mx(n)*hhy_o1+mx(n)*mz(n)*hhz_o1-(my(n)^2+mz(n)^2)*hhx_o1))-...
                                      dt(n)^2/(2*dt(n-1))*(-1/(1+alpha^2)*(my(n-1)*hhz_o2-mz(n-1)*hhy_o2)-...
                                      alpha/(1+alpha^2)*(my(n-1)*mx(n-1)*hhy_o2+mx(n-1)*mz(n-1)*hhz_o2-...
                                      (my(n-1)^2+mz(n-1)^2)*hhx_o2));
                        my_ab2=my(n)+(dt(n)+dt(n)^2/(2*dt(n-1)))*(-1/(1+alpha^2)*(mz(n)*hhx_o1-mx(n)*hhz_o1)-...
                                      alpha/(1+alpha^2)*(mx(n)*my(n)*hhx_o1+my(n)*mz(n)*hhz_o1-(mx(n)^2+mz(n)^2)*hhy_o1))-...
                                      dt(n)^2/(2*dt(n-1))*(-1/(1+alpha^2)*(mz(n-1)*hhx_o2-mx(n-1)*hhz_o2)-...
                                      alpha/(1+alpha^2)*(mx(n-1)*my(n-1)*hhx_o2+my(n-1)*mz(n-1)*hhz_o2-...
                                      (mx(n-1)^2+mz(n-1)^2)*hhy_o2));
                        mz_ab2=mz(n)+(dt(n)+dt(n)^2/(2*dt(n-1)))*(-1/(1+alpha^2)*(mx(n)*hhy_o1-my(n)*hhx_o1)-...
                                      alpha/(1+alpha^2)*(mx(n)*mz(n)*hhx_o1+my(n)*mz(n)*hhy_o1-(mx(n)^2+my(n)^2)*hhz_o1))-...
                                      dt(n)^2/(2*dt(n-1))*(-1/(1+alpha^2)*(mx(n-1)*hhy_o2-my(n-1)*hhx_o2)-...
                                      alpha/(1+alpha^2)*(mx(n-1)*mz(n-1)*hhx_o2+my(n-1)*mz(n-1)*hhy_o2-...
                                      (mx(n-1)^2+my(n-1)^2)*hhz_o2));  
                  end
                  m_ab2=[mx_ab2;my_ab2;mz_ab2];    
              %   TR corrector step
                  mx_n=mx(n); my_n=my(n); mz_n=mz(n);   %   Newton init. guess
                  m_n=[mx_n;my_n;mz_n];
                  hx_n=0; hy_n=0; hz_n=-1.1; 
                  hx=[hx,hx_n]; hy=[hy,hy_n]; hz=[hz,hz_n];
                  switch(field)
                     case(1)
                        rx=mx_n+dtc/2*(1/(1+alpha^2)*(my_n*hz(n+1)-mz_n*hy(n+1))+alpha/(1+alpha^2)*...
                                (mx_n*my_n*hy(n+1)+mx_n*mz_n*hz(n+1)-(my_n^2+mz_n^2)*hx(n)))-mx(n)+...
                                +dtc/2*(1/(1+alpha^2)*(my(n)*hz(n)-mz(n)*hy(n))+alpha/(1+alpha^2)*...
                                (mx(n)*my(n)*hy(n)+mx(n)*mz(n)*hz(n)-(my(n)^2+mz(n)^2)*hx(n)));
                        ry=my_n+dtc/2*(1/(1+alpha^2)*(mz_n*hx(n+1)-mx_n*hz(n+1))+alpha/(1+alpha^2)*...
                                (my_n*mx_n*hx(n+1)+my_n*mz_n*hz(n+1)-(mx_n^2+mz_n^2)*hy(n+1)))-my(n)+...
                                +dtc/2*(1/(1+alpha^2)*(mz(n)*hx(n)-mx(n)*hz(n))+alpha/(1+alpha^2)*...
                                (my(n)*mx(n)*hx(n)+my(n)*mz(n)*hz(n)-(mx(n)^2+mz(n)^2)*hy(n)));
                        rz=mz_n+dtc/2*(1/(1+alpha^2)*(mx_n*hy(n+1)-my_n*hx(n+1))+alpha/(1+alpha^2)*...
                                (mz_n*mx_n*hx(n+1)+mz_n*my_n*hy(n+1)-(mx_n^2+my_n^2)*hz(n+1)))-mz(n)+...
                                +dtc/2*(1/(1+alpha^2)*(mx(n)*hy(n)-my(n)*hx(n))+alpha/(1+alpha^2)*...
                                (mz(n)*mx(n)*hx(n)+mz(n)*my(n)*hy(n)-(mx(n)^2+my(n)^2)*hz(n))); 
                     case(2)
                        me_n=mx_n*ex+my_n*ey+mz_n*ez;
                        hhx_n=hx_n+k1*me_n*ex; hhy_n=hy_n+k1*me_n*ey; hhz_n=hz_n+k1*me_n*ez;
                        me_o=mx(n)*ex+my(n)*ey+mz(n)*ez;
                        hhx_o=hx(n)+k1*me_o*ex; hhy_o=hy(n)+k1*me_o*ey; hhz_o=hz(n)+k1*me_o*ez; 
                        rx=mx_n+dtc/2*(1/(1+alpha^2)*(my_n*hhz_n-mz_n*hhy_n)+alpha/(1+alpha^2)*...
                                (mx_n*my_n*hhy_n+mx_n*mz_n*hhz_n-(my_n^2+mz_n^2)*hhx_n))-mx(n)+...
                                +dtc/2*(1/(1+alpha^2)*(my(n)*hhz_o-mz(n)*hhy_o)+alpha/(1+alpha^2)*...
                                (mx(n)*my(n)*hhy_o+mx(n)*mz(n)*hhz_o-(my(n)^2+mz(n)^2)*hhx_o));
                        ry=my_n+dtc/2*(1/(1+alpha^2)*(mz_n*hhx_n-mx_n*hhz_n)+alpha/(1+alpha^2)*...
                                (my_n*mx_n*hhx_n+my_n*mz_n*hhz_n-(mx_n^2+mz_n^2)*hhy_n))-my(n)+...
                                +dtc/2*(1/(1+alpha^2)*(mz(n)*hhx_o-mx(n)*hhz_o)+alpha/(1+alpha^2)*...
                                (my(n)*mx(n)*hhx_o+my(n)*mz(n)*hhz_o-(mx(n)^2+mz(n)^2)*hhy_o));
                        rz=mz_n+dtc/2*(1/(1+alpha^2)*(mx_n*hhy_n-my_n*hhx_n)+alpha/(1+alpha^2)*...
                                (mz_n*mx_n*hhx_n+mz_n*my_n*hhy_n-(mx_n^2+my_n^2)*hhz_n))-mz(n)+...
                                +dtc/2*(1/(1+alpha^2)*(mx(n)*hhy_o-my(n)*hhx_o)+alpha/(1+alpha^2)*...
                                (mz(n)*mx(n)*hhx_o+mz(n)*my(n)*hhy_o-(mx(n)^2+my(n)^2)*hhz_o));  
                  end
                  r_vec=[rx;ry;rz]; r0_norm=norm(r_vec,2); n_new=0; r_norm=1;
                  fprintf('   ||r_nl(0)||_2=%10.4e\n',r0_norm);
                  while(r_norm>=max(1.e-9*r0_norm,1.2e-15))
                     switch(field)
                        case(1) 
                           j11=1+(alpha*dtc)/(2*(1+alpha^2))*(my_n*hy(n+1)+mz_n*hz(n+1));
                           j12=dtc/(2*(1+alpha^2))*hz(n+1)+(alpha*dtc)/(2*(1+alpha^2))*(mx_n*hy(n+1)-2*my_n*hx(n+1));
                           j13=-dtc/(2*(1+alpha^2))*hy(n+1)+(alpha*dtc)/(2*(1+alpha^2))*(mx_n*hz(n+1)-2*mz_n*hx(n+1));
                           j21=-dtc/(2*(1+alpha^2))*hz(n+1)+(alpha*dtc)/(2*(1+alpha^2))*(my_n*hx(n+1)-2*mx_n*hy(n+1));
                           j22=1+(alpha*dtc)/(2*(1+alpha^2))*(mx_n*hx(n+1)+mz_n*hz(n+1));
                           j23=dtc/(2*(1+alpha^2))*hx(n+1)+(alpha*dtc)/(2*(1+alpha^2))*(my_n*hz(n+1)-2*mz_n*hy(n+1));
                           j31=dtc/(2*(1+alpha^2))*hy(n+1)+(alpha*dtc)/(2*(1+alpha^2))*(mz_n*hx(n+1)-2*mx_n*hz(n+1));
                           j32=-dtc/(2*(1+alpha^2))*hy(n+1)+(alpha*dtc)/(2*(1+alpha^2))*(mz_n*hy(n+1)-2*my_n*hz(n+1));
                           j33=1+(alpha*dtc)/(2*(1+alpha^2))*(mx_n*hx(n+1)+my_n*hy(n+1));
                        case(2) 
                           j11=1+dtc/(2*(1+alpha^2))*(k1*(my_n*ex*ez-mz_n*ex*ey))+...
                                 (alpha*dtc)/(2*(1+alpha^2))*(my_n*hhy_n+k1*my_n*mx_n*ex*ey+mz_n*hhz_n+k1*mx_n*mz_n*ex*ez-...
                                 k1*(my_n^2+mz_n^2)*ex*ex);
                           j12=dtc/(2*(1+alpha^2))*(hhz_n+k1*my_n*ey*ez-k1*mz_n*ey*ey)+...
                               (alpha*dtc)/(2*(1+alpha^2))*(mx_n*hhy_n+k1*my_n*mx_n*ey*ey+k1*mx_n*mz_n*ey*ez-...
                                2*my_n*hhx_n-k1*(my_n^2+mz_n^2)*ey*ex);
                           j13=dtc/(2*(1+alpha^2))*(k1*my_n*ez*ez-hhy_n-k1*mz_n*ez*ey)+...
                               (alpha*dtc)/(2*(1+alpha^2))*(k1*my_n*mx_n*ez*ey+mx_n*hhz_n+k1*mx_n*mz_n*ez*ez-...
                                2*mz_n*hhx_n-k1*(my_n^2+mz_n^2)*ez*ex);
                           j21=dtc/(2*(1+alpha^2))*(k1*mz_n*ex*ex-hhz_n-k1*mx_n*ex*ez)+...
                               (alpha*dtc)/(2*(1+alpha^2))*(my_n*hhx_n+k1*mx_n*my_n*ex*ex+...
                               k1*my_n*mz_n*ex*ez-2*mx_n*hhy_n-k1*(mx_n^2+mz_n^2)*ex*ey);
                           j22=1+dtc/(2*(1+alpha^2))*(k1*(mz_n*ey*ex-mx_n*ey*ez))+...
                                 (alpha*dtc)/(2*(1+alpha^2))*(mx_n*hhx_n+k1*mx_n*my_n*ey*ex+mz_n*hhz_n+k1*my_n*mz_n*ey*ez-...
                                 k1*(mx_n^2+mz_n^2)*ey*ey);
                           j23=dtc/(2*(1+alpha^2))*(hhx_n+k1*mz_n*ez*ex-k1*mx_n*ez*ez)+...
                               (alpha*dtc)/(2*(1+alpha^2))*(k1*mx_n*my_n*ez*ex+my_n*hhz_n+k1*my_n*mz_n*ez*ez-...
                               2*mz_n*hhy_n-k1*(mx_n^2+mz_n^2)*ez*ey);
                           j31=dtc/(2*(1+alpha^2))*(hhy_n+k1*mx_n*ex*ey-k1*my_n*ex*ex)+...
                               (alpha*dtc)/(2*(1+alpha^2))*(mz_n*hhx_n+k1*mx_n*mz_n*ex*ex+k1*my_n*mz_n*ex*ey-...
                               2*mx_n*hhz_n-k1*(mx_n^2+my_n^2)*ex*ez);
                           j32=dtc/(2*(1+alpha^2))*(k1*mx_n*ey*ey-hhx_n-k1*my_n*ey*ex)+...
                               (alpha*dtc)/(2*(1+alpha^2))*(k1*mx_n*mz_n*ey*ex+mz_n*hhy_n+k1*my_n*mz_n*ey*ey-...
                               2*my_n*hhz_n-k1*(mx_n^2+my_n^2)*ey*ez);
                           j33=1+dtc/(2*(1+alpha^2))*(k1*(mx_n*ez*ey-my_n*ez*ex))+...
                                 (alpha*dtc)/(2*(1+alpha^2))*(mx_n*hhx_n+k1*mx_n*mz_n*ez*ex+my_n*hhy_n+k1*my_n*mz_n*ez*ey-...
                                 k1*(mx_n^2+my_n^2)*ez*ez); 
                     end
                     J=[j11,j12,j13;j21,j22,j23;j31,j32,j33];
                     dm_n=J\r_vec;
                     m_n=m_n-dm_n;
                     mx_n=m_n(1); my_n=m_n(2); mz_n=m_n(3);
                     switch(field)
                        case(1) 
                           rx=mx_n+dtc/2*(1/(1+alpha^2)*(my_n*hz(n+1)-mz_n*hy(n+1))+alpha/(1+alpha^2)*...
                                   (mx_n*my_n*hy(n+1)+mx_n*mz_n*hz(n+1)-(my_n^2+mz_n^2)*hx(n+1)))-mx(n)+...
                                   +dtc/2*(1/(1+alpha^2)*(my(n)*hz(n)-mz(n)*hy(n))+alpha/(1+alpha^2)*...
                                   (mx(n)*my(n)*hy(n)+mx(n)*mz(n)*hz(n)-(my(n)^2+mz(n)^2)*hx(n)));
                           ry=my_n+dtc/2*(1/(1+alpha^2)*(mz_n*hx(n+1)-mx_n*hz(n+1))+alpha/(1+alpha^2)*...
                                   (my_n*mx_n*hx(n+1)+my_n*mz_n*hz(n+1)-(mx_n^2+mz_n^2)*hy(n+1)))-my(n)+...
                                   +dtc/2*(1/(1+alpha^2)*(mz(n)*hx(n)-mx(n)*hz(n))+alpha/(1+alpha^2)*...
                                   (my(n)*mx(n)*hx(n)+my(n)*mz(n)*hz(n)-(mx(n)^2+mz(n)^2)*hy(n)));
                           rz=mz_n+dtc/2*(1/(1+alpha^2)*(mx_n*hy(n+1)-my_n*hx(n+1))+alpha/(1+alpha^2)*...
                                   (mz_n*mx_n*hx(n+1)+mz_n*my_n*hy(n+1)-(mx_n^2+my_n^2)*hz(n+1)))-mz(n)+...
                                   +dtc/2*(1/(1+alpha^2)*(mx(n)*hy(n)-my(n)*hx(n))+alpha/(1+alpha^2)*...
                                   (mz(n)*mx(n)*hx(n)+mz(n)*my(n)*hy(n)-(mx(n)^2+my(n)^2)*hz(n))); 
                        case(2)
                           me_n=mx_n*ex+my_n*ey+mz_n*ez;
                           hhx_n=hx_n+k1*me_n*ex; hhy_n=hy_n+k1*me_n*ey; hhz_n=hz_n+k1*me_n*ez;
                           rx=mx_n+dtc/2*(1/(1+alpha^2)*(my_n*hhz_n-mz_n*hhy_n)+alpha/(1+alpha^2)*...
                                   (mx_n*my_n*hhy_n+mx_n*mz_n*hhz_n-(my_n^2+mz_n^2)*hhx_n))-mx(n)+...
                                   +dtc/2*(1/(1+alpha^2)*(my(n)*hhz_o-mz(n)*hhy_o)+alpha/(1+alpha^2)*...
                                   (mx(n)*my(n)*hhy_o+mx(n)*mz(n)*hhz_o-(my(n)^2+mz(n)^2)*hhx_o));
                           ry=my_n+dtc/2*(1/(1+alpha^2)*(mz_n*hhx_n-mx_n*hhz_n)+alpha/(1+alpha^2)*...
                                   (my_n*mx_n*hhx_n+my_n*mz_n*hhz_n-(mx_n^2+mz_n^2)*hhy_n))-my(n)+...
                                   +dtc/2*(1/(1+alpha^2)*(mz(n)*hhx_o-mx(n)*hhz_o)+alpha/(1+alpha^2)*...
                                   (my(n)*mx(n)*hhx_o+my(n)*mz(n)*hhz_o-(mx(n)^2+mz(n)^2)*hhy_o));
                           rz=mz_n+dtc/2*(1/(1+alpha^2)*(mx_n*hhy_n-my_n*hhx_n)+alpha/(1+alpha^2)*...
                                   (mz_n*mx_n*hhx_n+mz_n*my_n*hhy_n-(mx_n^2+my_n^2)*hhz_n))-mz(n)+...
                                   +dtc/2*(1/(1+alpha^2)*(mx(n)*hhy_o-my(n)*hhx_o)+alpha/(1+alpha^2)*...
                                   (mz(n)*mx(n)*hhx_o+mz(n)*my(n)*hhy_o-(mx(n)^2+my(n)^2)*hhz_o));   
                     end
                     r_vec=[rx;ry;rz]; r_norm=norm(r_vec,2);
                     n_new=n_new+1;
                  end
                  fprintf('   Newton method converged in %1i steps\n',n_new);
                  fprintf('   ||r_nl(%2i)||_2=%10.4e\n',n_new,r_norm); 
                  mx=[mx,mx_n]; my=[my,my_n]; mz=[mz,mz_n];
                  if(field==2)
                     me=mx_n*ex+my_n*ey+mz_n*ez;
                     hmcx=[hmcx,k1*me*ex]; hmcy=[hmcy,k1*me*ey]; hmcz=[hmcz,k1*me*ez];
                  end
                  fprintf('   AB2:    mx(%8.4f)=%7.4f; my(%8.4f)=%7.4f; mz(%8.4f)=%7.4f\n',t_c,mx_ab2,t_c,my_ab2,t_c,mz_ab2);
                  fprintf(fid,'   AB2:    mx(%8.4f)=%7.4f; my(%8.4f)=%7.4f; mz(%8.4f)=%7.4f\n',t_c,mx_ab2,t_c,my_ab2,t_c,mz_ab2);
                  fprintf('   TR:  mx(%8.4f)=%7.4f; my(%8.4f)=%7.4f; mz(%8.4f)=%7.4f\n',t_c,mx_n,t_c,my_n,t_c,mz_n);
                  fprintf(fid,'   TR:  mx(%8.4f)=%7.4f; my(%8.4f)=%7.4f; mz(%8.4f)=%7.4f\n',t_c,mx_n,t_c,my_n,t_c,mz_n);
                  %   LTE estimate
                  if(fin==1) 
                     break;
                  end
                  d=1/(3*(1+dt(n-1)/dt(n)))*(m_n-m_ab2); dh=[dh,norm(d,2)];
                  fprintf('   ||d(%5i)||=%8.3e\n',n,norm(d,2));
                  fprintf(fid,'   ||d(%5i)||=%8.3e\n',n,norm(d,2));
                  dt_new=(eps_t/norm(d,2))^(1/3)*dt(n); 
                  if(t_c+dt_new>=t_end)
                     dt_new=abs(t_end-t_c);
                     t_c=t_end; fin=1;
                  else
                     t_c=t_c+dt_new;
                  end
                  dt=[dt,dt_new]; th=[th,t_c]; dtc=dt_new;
               end
               set(gcf,'Position',[575,240,700,700]);
               subplot(221); plot(th,mx,'r.-','MarkerSize',4); 
               axis([0 t_end -1.01 1.01]);
               title('x-component of the magnetization m');
               xlabel('t'); ylabel('mx');
               subplot(222); plot(th,my,'b.-','MarkerSize',4);
               axis([0 t_end -1.01 1.01]);
               title('y-component of the magnetization m');
               xlabel('t'); ylabel('my');
               subplot(223); plot(th,mz,'g.-','MarkerSize',4);
               axis([0 t_end -1.01 1.01]);
               title('z-component of the magnetization m');
               xlabel('t'); ylabel('mz');
               subplot(224); plot(th,[0,dt],'k.-','MarkerSize',4);
               axis([0 t_end 0 1.1*max(dt)]);
               title('Time step history');
               xlabel('t'); ylabel('dt');
           otherwise
              fprintf('Wrong integration mode\n');  
       end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                      eBDF3/AB3/RK3-IMR                                  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    case(3)   %  eBDF3/AB3/RK3-IMR
       switch(mode)
          case(1)   %   fixed step size
             t_c=dtc;   %   current time
             n=0;       %   step counter
             dt=[dtc];  %   time step history
             th=[0];    %   discrete time history
             fin=0;     %   finished flag 
             while(t_c<=t_end && fin==0)
                n=n+1;
                fprintf('STEP %5i, t=%8.3e, dt=%8.3e\n',n,t_c,dtc);
                fprintf(fid,'STEP %5i, t=%8.3e, dt=%8.3e\n',n,t_c,dtc);
                mx_n=mx(n); my_n=my(n); mz_n=mz(n);   %   Newton init. guess
                m_n=[mx_n;my_n;mz_n];
                hx_n=0; hy_n=0; hz_n=-1.1;   %  external field
                hx=[hx,hx_n]; hy=[hy,hy_n]; hz=[hz,hz_n];
                mx_h=(mx_n+mx(n))/2; my_h=(my_n+my(n))/2; mz_h=(mz_n+mz(n))/2;
                hx_h=(hx(n+1)+hx(n))/2; hy_h=(hy(n+1)+hy(n))/2; hz_h=(hz(n+1)+hz(n))/2;
                switch(field)
                   case(1)
                      rx=mx_n+dtc*(1/(1+alpha^2)*(my_h*hz_h-mz_h*hy_h)+alpha/(1+alpha^2)*...
                                  (my_h*mx_h*hy_h+mx_h*mz_h*hz_h-(my_h^2+mz_h^2)*hx_h))-mx(n);
                      ry=my_n+dtc*(1/(1+alpha^2)*(mz_h*hx_h-mx_h*hz_h)+alpha/(1+alpha^2)*...
                                  (mx_h*my_h*hx_h+my_h*mz_h*hz_h-(mx_h^2+mz_h^2)*hy_h))-my(n);
                      rz=mz_n+dtc*(1/(1+alpha^2)*(mx_h*hy_h-my_h*hx_h)+alpha/(1+alpha^2)*...
                                  (mx_h*mz_h*hx_h+my_h*mz_h*hy_h-(mx_h^2+my_h^2)*hz_h))-mz(n); 
                   case(2)
                      me_h=(mx_n+mx(n))/2*ex+(my_n+my(n))/2*ey+(mz_n+mz(n))/2*ez;
                      hhx_h=hx_h+k1*me_h*ex; hhy_h=hy_h+k1*me_h*ey; hhz_h=hz_h+k1*me_h*ez;
                      rx=mx_n+dtc*(1/(1+alpha^2)*(my_h*hhz_h-mz_h*hhy_h)+alpha/(1+alpha^2)*...
                                  (my_h*mx_h*hhy_h+mx_h*mz_h*hhz_h-(my_h^2+mz_h^2)*hhx_h))-mx(n);
                      ry=my_n+dtc*(1/(1+alpha^2)*(mz_h*hhx_h-mx_h*hhz_h)+alpha/(1+alpha^2)*...
                                  (mx_h*my_h*hhx_h+my_h*mz_h*hhz_h-(mx_h^2+mz_h^2)*hhy_h))-my(n);
                      rz=mz_n+dtc*(1/(1+alpha^2)*(mx_h*hhy_h-my_h*hhx_h)+alpha/(1+alpha^2)*...
                                  (mx_h*mz_h*hhx_h+my_h*mz_h*hhy_h-(mx_h^2+my_h^2)*hhz_h))-mz(n); 
                end
                r_vec=[rx;ry;rz]; r0_norm=norm(r_vec,2); n_new=0; r_norm=1;
                fprintf('   ||r_nl(0)||_2=%10.4e\n',r0_norm);
                while(r_norm>=max(1.e-9*r0_norm,1.2e-15))
                   switch(field)
                      case(1)
                         j11=1+(alpha*dtc)/(2*(1+alpha^2))*(my_h*hy_h+mz_h*hz_h);
                         j12=dtc/(2*(1+alpha^2))*hz_h+(alpha*dtc)/(2*(1+alpha^2))*(mx_h*hy_h-2*my_h*hx_h);
                         j13=-dtc/(2*(1+alpha^2))*hy_h+(alpha*dtc)/(2*(1+alpha^2))*(mx_h*hz_h-2*mz_h*hx_h);
                         j21=-dtc/(2*(1+alpha^2))*hz_h+(alpha*dtc)/(2*(1+alpha^2))*(my_h*hx_h-2*mx_h*hy_h);
                         j22=1+(alpha*dtc)/(2*(1+alpha^2))*(mx_h*hx_h+mz_h*hz_h);
                         j23=dtc/(2*(1+alpha^2))*hx_h+(alpha*dtc)/(2*(1+alpha^2))*(my_h*hz_h-2*mz_h*hy_h);
                         j31=dtc/(2*(1+alpha^2))*hy_h+(alpha*dtc)/(2*(1+alpha^2))*(mz_h*hx_h-2*mx_h*hz_h);
                         j32=-dtc/(2*(1+alpha^2))*hx_h+(alpha*dtc)/(2*(1+alpha^2))*(mz_h*hy_h-2*my_h*hz_h);
                         j33=1+(alpha*dtc)/(2*(1+alpha^2))*(mx_h*hx_h+my_h*hy_h);
                      case(2)
                         j11=1+(dtc/(2*(1+alpha^2)))*(k1*(my_h*ex*ez-mz_h*ex*ey))+(dtc*alpha)/(2*(1+alpha^2))*...
                               (my_h*hhy_h+k1*my_h*mx_h*ex*ey+mz_h*hhz_h+k1*mx_h*mz_h*ex*ez-k1*(my_h^2+mz_h^2)*ex*ex);
                         j12=dtc/(2*(1+alpha^2))*(hhz_h-k1*mz_h*ey*ey+k1*my_h*ey*ez)+(dtc*alpha)/(2*(1+alpha^2))*...
                               (mx_h*hy_h+k1*mx_h*my_h*ey*ey+k1*mx_h*mz_h*ey*ez-2*my_h*hhx_h-k1*(my_h^2+mz_h^2)*ey*ex);
                         j13=dtc/(2*(1+alpha^2))*(k1*my_h*ez*ez-hhy_h-k1*mz_h*ez*ey)+(dtc*alpha)/(2*(1+alpha^2))*...
                               (k1*my_h*mx_h*ez*ey+mx_h*hhz_h+k1*mx_h*mz_h*ez*ez-2*mz_h*hhx_h-k1*(my_h^2+mz_h^2)*ez*ex);
                         j21=dtc/(2*(1+alpha^2))*(k1*mz_h*ex*ex-hhz_h-k1*mx_h*ex*ez)+(dtc*alpha)/(2*(1+alpha^2))*...
                               (my_h*hhx_h+k1*mx_h*my_h*ex*ex+k1*my_h*mz_h*ex*ez-2*mx_h*hhy_h-k1*(mx_h^2+mz_h^2)*ex*ey);
                         j22=1+(dtc/(2*(1+alpha^2)))*(k1*(mz_h*ey*ex-mx_h*ey*ez))+(dtc*alpha)/(2*(1+alpha^2))*...
                               (mx_h*hx_h+k1*mx_h*my_h*ey*ex+mz_h*hhz_h+k1*my_h*mz_h*ey*ez-k1*(mx_h^2+mz_h^2)*ey*ey);
                         j23=dtc/(2*(1+alpha^2))*(hhx_h+k1*mz_h*ez*ex-k1*mx_h*ez*ez)+(dtc*alpha)/(2*(1+alpha^2))*...
                               (k1*mx_h*my_h*ez*ex+my_h*hhz_h+k1*my_h*mz_h*ez*ez-2*mz_h*hhy_h-k1*(mx_h^2+mz_h^2)*ez*ey);
                         j31=dtc/(2*(1+alpha^2))*(hhy_h-k1*my_h*ex*ex+k1*mx_h*ex*ey)+(dtc*alpha)/(2*(1+alpha^2))*...
                               (mz_h*hhx_h+k1*mx_h*mz_h*ex*ex+k1*my_h*mz_h*ex*ey-2*mx_h*hhz_h-k1*(mx_h^2+my_h^2)*ex*ez);
                         j32=dtc/(2*(1+alpha^2))*(k1*mx_h*ey*ey-hhx_h-k1*my_h*ey*ex)+(dtc*alpha)/(2*(1+alpha^2))*...
                               (k1*mx_h*mz_h*ey*ex+mz_h*hhy_h+k1*my_h*mz_h*ey*ey-2*my_h*hhz_h-k1*(mx_h^2+my_h^2)*ey*ez);
                         j33=1+(dtc/(2*(1+alpha^2)))*(k1*(mx_h*ez*ey-my_h*ez*ex))+(dtc*alpha)/(2*(1+alpha^2))*...
                               (mx_h*hhx_h+k1*mx_h*mz_h*ez*ex+my_h*hhy_h+k1*my_h*mz_h*ez*ey-k1*(mx_h^2+my_h)^2*ez*ez); 
                   end
                   J=[j11,j12,j13;j21,j22,j23;j31,j32,j33];
                   dm_n=J\r_vec;
                   m_n=m_n-dm_n;
                   mx_n=m_n(1); my_n=m_n(2); mz_n=m_n(3);
                   mx_h=(mx_n+mx(n))/2; my_h=(my_n+my(n))/2; mz_h=(mz_n+mz(n))/2;
                   switch(field)
                      case(1)
                         rx=mx_n+dtc*(1/(1+alpha^2)*(my_h*hz_h-mz_h*hy_h)+alpha/(1+alpha^2)*...
                                     (my_h*mx_h*hy_h+mx_h*mz_h*hz_h-(my_h^2+mz_h^2)*hx_h))-mx(n);
                         ry=my_n+dtc*(1/(1+alpha^2)*(mz_h*hx_h-mx_h*hz_h)+alpha/(1+alpha^2)*...
                                     (mx_h*my_h*hx_h+my_h*mz_h*hz_h-(mx_h^2+mz_h^2)*hy_h))-my(n);
                         rz=mz_n+dtc*(1/(1+alpha^2)*(mx_h*hy_h-my_h*hx_h)+alpha/(1+alpha^2)*...
                                     (mx_h*mz_h*hx_h+my_h*mz_h*hy_h-(mx_h^2+my_h^2)*hz_h))-mz(n);
                      case(2)
                         me_h=(mx_n+mx(n))/2*ex+(my_n+my(n))/2*ey+(mz_n+mz(n))/2*ez;
                         hhx_h=hx_h+k1*me_h*ex; hhy_h=hy_h+k1*me_h*ey; hhz_h=hz_h+k1*me_h*ez;
                         rx=mx_n+dtc*(1/(1+alpha^2)*(my_h*hhz_h-mz_h*hhy_h)+alpha/(1+alpha^2)*...
                                     (my_h*mx_h*hhy_h+mx_h*mz_h*hhz_h-(my_h^2+mz_h^2)*hhx_h))-mx(n);
                         ry=my_n+dtc*(1/(1+alpha^2)*(mz_h*hhx_h-mx_h*hhz_h)+alpha/(1+alpha^2)*...
                                     (mx_h*my_h*hhx_h+my_h*mz_h*hhz_h-(mx_h^2+mz_h^2)*hhy_h))-my(n);
                         rz=mz_n+dtc*(1/(1+alpha^2)*(mx_h*hhy_h-my_h*hhx_h)+alpha/(1+alpha^2)*...
                                     (mx_h*mz_h*hhx_h+my_h*mz_h*hhy_h-(mx_h^2+my_h^2)*hhz_h))-mz(n);  
                   end
                   r_vec=[rx;ry;rz]; r_norm=norm(r_vec,2);
                   n_new=n_new+1;
                end
                fprintf('   Newton method converged in %1i steps\n',n_new);
                fprintf('   ||r_nl(%2i)||_2=%10.4e\n',n_new,r_norm); 
                mx=[mx,mx_n]; my=[my,my_n]; mz=[mz,mz_n];
                if(field==2)
                   me=mx_n*ex+my_n*ey+mz_n*ez;
                   hmcx=[hmcx,k1*me*ex]; hmcy=[hmcy,k1*me*ey]; hmcz=[hmcz,k1*me*ez];
                end
                fprintf('   IMR:  mx(%8.4f)=%7.4f; my(%8.4f)=%7.4f; mz(%8.4f)=%7.4f\n',t_c,mx_n,t_c,my_n,t_c,mz_n);
                fprintf(fid,'   IMR:  mx(%8.4f)=%7.4f; my(%8.4f)=%7.4f; mz(%8.4f)=%7.4f\n',t_c,mx_n,t_c,my_n,t_c,mz_n);
                if(t_c==t_end)
                   fin=1;
                end
                t_c=t_c+dtc;
                if(t_c>=t_end)
                   dtc=dtc-t_c+t_end; 
                   t_c=t_end; 
                   if(dtc<=1.e-6)
                        fin=1;
                   end
                end
                dt=[dt,dtc]; th=[th,t_c]; 
             end
             set(gcf,'Position',[575,240,700,700]);
             subplot(221); plot(th,mx,'r.-','MarkerSize',4); 
             axis([0 t_end -1.01 1.01]);
             title('x-component of the magnetization m');
             xlabel('t'); ylabel('mx');
             subplot(222); plot(th,my,'b.-','MarkerSize',4);
             axis([0 t_end -1.01 1.01]);
             title('y-component of the magnetization m');
             xlabel('t'); ylabel('my');
             subplot(223); plot(th,mz,'g.-','MarkerSize',4);
             axis([0 t_end -1.01 1.01]);
             title('z-component of the magnetization m');
             xlabel('t'); ylabel('mz');
             subplot(224); plot(th,dt,'k.-','MarkerSize',4);
             axis([0 t_end 0 1.1*max(dt)]);
             title('Time step history');
             xlabel('t'); ylabel('dt'); 
          case(2)   %   variable step size   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
             dt=[dt0];   %  the time step history
             t_c=dt0;    %  current time
             dtc=dt0;    %  current time step
             th=[0];     %  time history
             dh=[0,0,0]; %  LTE estimate history
             dexh=[0,0,0];  %  exact LTE history
             %  Startup (2 steps of IMR with dt(0))
             for n=1:2 
                fprintf('STEP %5i, t=%8.3e\n',n,t_c);
                fprintf(fid,'STEP %5i, t=%8.3e\n',n,t_c);
                mx_n=mx(n); my_n=my(n); mz_n=mz(n);   %   Newton init. guess
                m_n=[mx_n;my_n;mz_n];
                hx_n=0; hy_n=0; hz_n=-1.1;   %  external field
                hx=[hx,hx_n]; hy=[hy,hy_n]; hz=[hz,hz_n];
                mx_h=(mx_n+mx(n))/2; my_h=(my_n+my(n))/2; mz_h=(mz_n+mz(n))/2;
                hx_h=(hx(n+1)+hx(n))/2; hy_h=(hy(n+1)+hy(n))/2; hz_h=(hz(n+1)+hz(n))/2;
                switch(field)
                   case(1) 
                      rx=mx_n+dtc*(1/(1+alpha^2)*(my_h*hz_h-mz_h*hy_h)+alpha/(1+alpha^2)*...
                                  (my_h*mx_h*hy_h+mx_h*mz_h*hz_h-(my_h^2+mz_h^2)*hx_h))-mx(n);
                      ry=my_n+dtc*(1/(1+alpha^2)*(mz_h*hx_h-mx_h*hz_h)+alpha/(1+alpha^2)*...
                                  (mx_h*my_h*hx_h+my_h*mz_h*hz_h-(mx_h^2+mz_h^2)*hy_h))-my(n);
                      rz=mz_n+dtc*(1/(1+alpha^2)*(mx_h*hy_h-my_h*hx_h)+alpha/(1+alpha^2)*...
                                  (mx_h*mz_h*hx_h+my_h*mz_h*hy_h-(mx_h^2+my_h^2)*hz_h))-mz(n); 
                   case(2)
                      me_h=(mx_n+mx(n))/2*ex+(my_n+my(n))/2*ey+(mz_n+mz(n))/2*ez;
                      hhx_h=hx_h+k1*me_h*ex; hhy_h=hy_h+k1*me_h*ey; hhz_h=hz_h+k1*me_h*ez;
                      rx=mx_n+dtc*(1/(1+alpha^2)*(my_h*hhz_h-mz_h*hhy_h)+alpha/(1+alpha^2)*...
                                  (my_h*mx_h*hhy_h+mx_h*mz_h*hhz_h-(my_h^2+mz_h^2)*hhx_h))-mx(n);
                      ry=my_n+dtc*(1/(1+alpha^2)*(mz_h*hhx_h-mx_h*hhz_h)+alpha/(1+alpha^2)*...
                                  (mx_h*my_h*hhx_h+my_h*mz_h*hhz_h-(mx_h^2+mz_h^2)*hhy_h))-my(n);
                      rz=mz_n+dtc*(1/(1+alpha^2)*(mx_h*hhy_h-my_h*hhx_h)+alpha/(1+alpha^2)*...
                                  (mx_h*mz_h*hhx_h+my_h*mz_h*hhy_h-(mx_h^2+my_h^2)*hhz_h))-mz(n);  
                end
                r_vec=[rx;ry;rz]; r0_norm=norm(r_vec,2); n_new=0; r_norm=1;
                fprintf('   ||r_nl(0)||_2=%10.4e\n',r0_norm);
                while(r_norm>=max(1.e-9*r0_norm,1.2e-15))
                   switch(field)
                      case(1)
                         j11=1+(alpha*dtc)/(2*(1+alpha^2))*(my_h*hy_h+mz_h*hz_h);
                         j12=dtc/(2*(1+alpha^2))*hz_h+(alpha*dtc)/(2*(1+alpha^2))*(mx_h*hy_h-2*my_h*hx_h);
                         j13=-dtc/(2*(1+alpha^2))*hy_h+(alpha*dtc)/(2*(1+alpha^2))*(mx_h*hz_h-2*mz_h*hx_h);
                         j21=-dtc/(2*(1+alpha^2))*hz_h+(alpha*dtc)/(2*(1+alpha^2))*(my_h*hx_h-2*mx_h*hy_h);
                         j22=1+(alpha*dtc)/(2*(1+alpha^2))*(mx_h*hx_h+mz_h*hz_h);
                         j23=dtc/(2*(1+alpha^2))*hx_h+(alpha*dtc)/(2*(1+alpha^2))*(my_h*hz_h-2*mz_h*hy_h);
                         j31=dtc/(2*(1+alpha^2))*hy_h+(alpha*dtc)/(2*(1+alpha^2))*(mz_h*hx_h-2*mx_h*hz_h);
                         j32=-dtc/(2*(1+alpha^2))*hx_h+(alpha*dtc)/(2*(1+alpha^2))*(mz_h*hy_h-2*my_h*hz_h);
                         j33=1+(alpha*dtc)/(2*(1+alpha^2))*(mx_h*hx_h+my_h*hy_h);
                      case(2)
                         j11=1+(dtc/(2*(1+alpha^2)))*(k1*(my_h*ex*ez-mz_h*ex*ey))+(dtc*alpha)/(2*(1+alpha^2))*...
                               (my_h*hhy_h+k1*my_h*mx_h*ex*ey+mz_h*hhz_h+k1*mx_h*mz_h*ex*ez-k1*(my_h^2+mz_h^2)*ex*ex);
                         j12=dtc/(2*(1+alpha^2))*(hhz_h-k1*mz_h*ey*ey+k1*my_h*ey*ez)+(dtc*alpha)/(2*(1+alpha^2))*...
                               (mx_h*hy_h+k1*mx_h*my_h*ey*ey+k1*mx_h*mz_h*ey*ez-2*my_h*hhx_h-k1*(my_h^2+mz_h^2)*ey*ex);
                         j13=dtc/(2*(1+alpha^2))*(k1*my_h*ez*ez-hhy_h-k1*mz_h*ez*ey)+(dtc*alpha)/(2*(1+alpha^2))*...
                               (k1*my_h*mx_h*ez*ey+mx_h*hhz_h+k1*mx_h*mz_h*ez*ez-2*mz_h*hhx_h-k1*(my_h^2+mz_h^2)*ez*ex);
                         j21=dtc/(2*(1+alpha^2))*(k1*mz_h*ex*ex-hhz_h-k1*mx_h*ex*ez)+(dtc*alpha)/(2*(1+alpha^2))*...
                               (my_h*hhx_h+k1*mx_h*my_h*ex*ex+k1*my_h*mz_h*ex*ez-2*mx_h*hhy_h-k1*(mx_h^2+mz_h^2)*ex*ey);
                         j22=1+(dtc/(2*(1+alpha^2)))*(k1*(mz_h*ey*ex-mx_h*ey*ez))+(dtc*alpha)/(2*(1+alpha^2))*...
                               (mx_h*hx_h+k1*mx_h*my_h*ey*ex+mz_h*hhz_h+k1*my_h*mz_h*ey*ez-k1*(mx_h^2+mz_h^2)*ey*ey);
                         j23=dtc/(2*(1+alpha^2))*(hhx_h+k1*mz_h*ez*ex-k1*mx_h*ez*ez)+(dtc*alpha)/(2*(1+alpha^2))*...
                               (k1*mx_h*my_h*ez*ex+my_h*hhz_h+k1*my_h*mz_h*ez*ez-2*mz_h*hhy_h-k1*(mx_h^2+mz_h^2)*ez*ey);
                         j31=dtc/(2*(1+alpha^2))*(hhy_h-k1*my_h*ex*ex+k1*mx_h*ex*ey)+(dtc*alpha)/(2*(1+alpha^2))*...
                               (mz_h*hhx_h+k1*mx_h*mz_h*ex*ex+k1*my_h*mz_h*ex*ey-2*mx_h*hhz_h-k1*(mx_h^2+my_h^2)*ex*ez);
                         j32=dtc/(2*(1+alpha^2))*(k1*mx_h*ey*ey-hhx_h-k1*my_h*ey*ex)+(dtc*alpha)/(2*(1+alpha^2))*...
                               (k1*mx_h*mz_h*ey*ex+mz_h*hhy_h+k1*my_h*mz_h*ey*ey-2*my_h*hhz_h-k1*(mx_h^2+my_h^2)*ey*ez);
                         j33=1+(dtc/(2*(1+alpha^2)))*(k1*(mx_h*ez*ey-my_h*ez*ex))+(dtc*alpha)/(2*(1+alpha^2))*...
                               (mx_h*hhx_h+k1*mx_h*mz_h*ez*ex+my_h*hhy_h+k1*my_h*mz_h*ez*ey-k1*(mx_h^2+my_h)^2*ez*ez);  
                   end
                   J=[j11,j12,j13;j21,j22,j23;j31,j32,j33];
                   dm_n=J\r_vec;
                   m_n=m_n-dm_n;
                   mx_n=m_n(1); my_n=m_n(2); mz_n=m_n(3);
                   mx_h=(mx_n+mx(n))/2; my_h=(my_n+my(n))/2; mz_h=(mz_n+mz(n))/2;
                   switch(field)
                      case(1)
                         rx=mx_n+dtc*(1/(1+alpha^2)*(my_h*hz_h-mz_h*hy_h)+alpha/(1+alpha^2)*...
                                     (my_h*mx_h*hy_h+mx_h*mz_h*hz_h-(my_h^2+mz_h^2)*hx_h))-mx(n);
                         ry=my_n+dtc*(1/(1+alpha^2)*(mz_h*hx_h-mx_h*hz_h)+alpha/(1+alpha^2)*...
                                     (mx_h*my_h*hx_h+my_h*mz_h*hz_h-(mx_h^2+mz_h^2)*hy_h))-my(n);
                         rz=mz_n+dtc*(1/(1+alpha^2)*(mx_h*hy_h-my_h*hx_h)+alpha/(1+alpha^2)*...
                                     (mx_h*mz_h*hx_h+my_h*mz_h*hy_h-(mx_h^2+my_h^2)*hz_h))-mz(n);
                      case(2)
                         me_h=(mx_n+mx(n))/2*ex+(my_n+my(n))/2*ey+(mz_n+mz(n))/2*ez;
                         hhx_h=hx_h+k1*me_h*ex; hhy_h=hy_h+k1*me_h*ey; hhz_h=hz_h+k1*me_h*ez;
                         rx=mx_n+dtc*(1/(1+alpha^2)*(my_h*hhz_h-mz_h*hhy_h)+alpha/(1+alpha^2)*...
                                     (my_h*mx_h*hhy_h+mx_h*mz_h*hhz_h-(my_h^2+mz_h^2)*hhx_h))-mx(n);
                         ry=my_n+dtc*(1/(1+alpha^2)*(mz_h*hhx_h-mx_h*hhz_h)+alpha/(1+alpha^2)*...
                                     (mx_h*my_h*hhx_h+my_h*mz_h*hhz_h-(mx_h^2+mz_h^2)*hhy_h))-my(n);
                         rz=mz_n+dtc*(1/(1+alpha^2)*(mx_h*hhy_h-my_h*hhx_h)+alpha/(1+alpha^2)*...
                                     (mx_h*mz_h*hhx_h+my_h*mz_h*hhy_h-(mx_h^2+my_h^2)*hhz_h))-mz(n);   
                   end
                   r_vec=[rx;ry;rz]; r_norm=norm(r_vec,2);
                   n_new=n_new+1;
                end
                fprintf('   Newton method converged in %1i steps\n',n_new);
                fprintf('   ||r_nl(%2i)||_2=%10.4e\n',n_new,r_norm); 
                mx=[mx,mx_n]; my=[my,my_n]; mz=[mz,mz_n];
                if(field==2)
                   me=mx_n*ex+my_n*ey+mz_n*ez;
                   hmcx=[hmcx,k1*me*ex]; hmcy=[hmcy,k1*me*ey]; hmcz=[hmcz,k1*me*ez];
                end
                fprintf('   IMR:  mx(%8.4f)=%7.4f; my(%8.4f)=%7.4f; mz(%8.4f)=%7.4f\n',t_c,mx_n,t_c,my_n,t_c,mz_n);
                fprintf(fid,'   IMR:  mx(%8.4f)=%7.4f; my(%8.4f)=%7.4f; mz(%8.4f)=%7.4f\n',t_c,mx_n,t_c,my_n,t_c,mz_n);
                dtc=3*dtc; dt=[dt,dtc]; th=[th,t_c]; t_c=t_c+dtc; 
             end 
             %  Startup for AB3 (2 steps of RK3 method with dt(0))
             for n=1:2 
                switch(field)
                   case(1)
                      kx1=-1/(1+alpha^2)*(my(n)*hz(n)-mz(n)*hy(n))-alpha/(1+alpha^2)*...
                          (my(n)*mx(n)*hy(n)+mx(n)*mz(n)*hz(n)-(my(n)^2+mz(n)^2)*hx(n));
                      ky1=-1/(1+alpha^2)*(mz(n)*hx(n)-mx(n)*hz(n))-alpha/(1+alpha^2)*...
                          (mx(n)*my(n)*hx(n)+my(n)*mz(n)*hz(n)-(mx(n)^2+mz(n)^2)*hy(n));
                      kz1=-1/(1+alpha^2)*(mx(n)*hy(n)-my(n)*hx(n))-alpha/(1+alpha^2)*...
                          (mx(n)*mz(n)*hx(n)+my(n)*mz(n)*hy(n)-(mx(n)^2+my(n)^2)*hz(n));
                      mx2=mx(n)+0.5*dt(n)*kx1; my2=my(n)+0.5*dt(n)*ky1; mz2=mz(n)+0.5*dt(n)*kz1;
                      kx2=-1/(1+alpha^2)*(my2*hz(n)-mz2*hy(n))-alpha/(1+alpha^2)*...
                          (my2*mx2*hy(n)+mx2*mz2*hz(n)-(my2^2+mz2^2)*hx(n));
                      ky2=-1/(1+alpha^2)*(mz2*hx(n)-mx2*hz(n))-alpha/(1+alpha^2)*...
                          (mx2*my2*hx(n)+my2*mz2*hz(n)-(mx2^2+mz2^2)*hy(n));
                      kz2=-1/(1+alpha^2)*(mx2*hy(n)-my2*hx(n))-alpha/(1+alpha^2)*...
                          (mx2*mz2*hx(n)+my2*mz2*hy(n)-(mx2^2+my2^2)*hz(n));
                      mx3=mx(n)+dt(n)*(-kx1+2*kx2);
                      my3=my(n)+dt(n)*(-ky1+2*ky2);
                      mz3=mz(n)+dt(n)*(-kz1+2*kz2);
                      kx3=-1/(1+alpha^2)*(my3*hz(n)-mz3*hy(n))-alpha/(1+alpha^2)*...
                          (my3*mx3*hy(n)+mx3*mz3*hz(n)-(my3^2+mz3^2)*hx(n));
                      ky3=-1/(1+alpha^2)*(mz3*hx(n)-mx3*hz(n))-alpha/(1+alpha^2)*...
                          (mx3*my3*hx(n)+my3*mz3*hz(n)-(mx3^2+mz3^2)*hy(n));
                      kz3=-1/(1+alpha^2)*(mx3*hy(n)-my3*hx(n))-alpha/(1+alpha^2)*...
                          (mx3*mz3*hx(n)+my3*mz3*hy(n)-(mx3^2+my3^2)*hz(n));
                    case(2)
                      me1=mx(n)*ex+my(n)*ey+mz(n)*ez;
                      hhx_1=hx(n)+k1*me1*ex; hhy_1=hy(n)+k1*me1*ey; hhz_1=hz(n)+k1*me1*ez;  
                      kx1=-1/(1+alpha^2)*(my(n)*hhz_1-mz(n)*hhy_1)-alpha/(1+alpha^2)*...
                          (my(n)*mx(n)*hhy_1+mx(n)*mz(n)*hhz_1-(my(n)^2+mz(n)^2)*hhx_1);
                      ky1=-1/(1+alpha^2)*(mz(n)*hhx_1-mx(n)*hhz_1)-alpha/(1+alpha^2)*...
                          (mx(n)*my(n)*hhx_1+my(n)*mz(n)*hhz_1-(mx(n)^2+mz(n)^2)*hhy_1);
                      kz1=-1/(1+alpha^2)*(mx(n)*hhy_1-my(n)*hhx_1)-alpha/(1+alpha^2)*...
                          (mx(n)*mz(n)*hhx_1+my(n)*mz(n)*hhy_1-(mx(n)^2+my(n)^2)*hhz_1);
                      mx2=mx(n)+0.5*dt(n)*kx1; my2=my(n)+0.5*dt(n)*ky1; mz2=mz(n)+0.5*dt(n)*kz1;
                      me2=mx2*ex+my2*ey+mz2*ez;
                      hhx_2=hx(n)+k1*me2*ex; hhy_2=hy(n)+k1*me2*ey; hhz_2=hz(n)+k1*me2*ez; 
                      kx2=-1/(1+alpha^2)*(my2*hhz_2-mz2*hhy_2)-alpha/(1+alpha^2)*...
                          (my2*mx2*hhy_2+mx2*mz2*hhz_2-(my2^2+mz2^2)*hhx_2);
                      ky2=-1/(1+alpha^2)*(mz2*hhx_2-mx2*hhz_2)-alpha/(1+alpha^2)*...
                          (mx2*my2*hhx_2+my2*mz2*hhz_2-(mx2^2+mz2^2)*hhy_2);
                      kz2=-1/(1+alpha^2)*(mx2*hhy_2-my2*hhx_2)-alpha/(1+alpha^2)*...
                          (mx2*mz2*hhx_2+my2*mz2*hhy_2-(mx2^2+my2^2)*hhz_2);
                      mx3=mx(n)+dt(n)*(-kx1+2*kx2);
                      my3=my(n)+dt(n)*(-ky1+2*ky2);
                      mz3=mz(n)+dt(n)*(-kz1+2*kz2);
                      me3=mx3*ex+my3*ey+mz3*ez;
                      hhx_3=hx(n)+k1*me3*ex; hhy_3=hy(n)+k1*me3*ey; hhz_3=hz(n)+k1*me3*ez;
                      kx3=-1/(1+alpha^2)*(my3*hhz_3-mz3*hhy_3)-alpha/(1+alpha^2)*...
                          (my3*mx3*hhy_3+mx3*mz3*hhz_3-(my3^2+mz3^2)*hhx_3);
                      ky3=-1/(1+alpha^2)*(mz3*hhx_3-mx3*hhz_3)-alpha/(1+alpha^2)*...
                          (mx3*my3*hhx_3+my3*mz3*hhz_3-(mx3^2+mz3^2)*hhy_3);
                      kz3=-1/(1+alpha^2)*(mx3*hhy_3-my3*hhx_3)-alpha/(1+alpha^2)*...
                          (mx3*mz3*hhx_3+my3*mz3*hhy_3-(mx3^2+my3^2)*hhz_3);      
                end
                mx_rk3=mx(n)+dt(n)/6*(kx1+4*kx2+kx3);
                my_rk3=my(n)+dt(n)/6*(ky1+4*ky2+ky3);
                mz_rk3=mz(n)+dt(n)/6*(kz1+4*kz2+kz3);
                mx_pred=[mx_pred,mx_rk3]; 
                my_pred=[my_pred,my_rk3];
                mz_pred=[mz_pred,mz_rk3];
             end
             %  General step 
             fin=0; th=[th,t_c];
             while(t_c<=t_end)
                n=n+1;
                fprintf('STEP %5i, t=%8.3e, dt=%8.3e\n',n,t_c,dt(n));
                fprintf('-------------------------------------\n');
                fprintf(fid,'STEP %5i, t=%8.3e, dt=%8.3e\n',n,t_c,dt(n));
                fprintf(fid,'-------------------------------------\n');
                switch(predict)  %  IMR predictors
                   case(1)   %   eBDF3 
                      mx_p0=mx(n); my_p0=my(n); mz_p0=mz(n);
                      mx_p1=mx(n-1); my_p1=my(n-1); mz_p1=mz(n-1);
                      mx_p2=mx(n-2); my_p2=my(n-2); mz_p2=mz(n-2);
                      switch(field)
                         case(1)
                            hx_p0=hx(n); hy_p0=hy(n); hz_p0=hz(n);
                            hx_p1=hx(n-1); hy_p1=hy(n-1); hz_p1=hz(n-1);
                            hx_p2=hx(n-2); hy_p2=hy(n-2); hz_p2=hz(n-2);
                         case(2)
                            me_0=mx(n)*ex+my(n)*ey+mz(n)*ez;
                            me_1=mx(n-1)*ex+my(n-1)*ey+mz(n-1)*ez;
                            me_2=mx(n-2)*ex+my(n-2)*ey+mz(n-2)*ez;
                            hx_p0=hx(n)+k1*me_0*ex; hy_p0=hy(n)+k1*me_0*ey; hz_p0=hz(n)+k1*me_0*ez;
                            hx_p1=hx(n-1)+k1*me_1*ex; hy_p1=hy(n-1)+k1*me_1*ey; hz_p1=hz(n-1)+k1*me_1*ez;
                            hx_p2=hx(n-2)+k1*me_2*ex; hy_p2=hy(n-2)+k1*me_2*ey; hz_p2=hz(n-2)+k1*me_2*ez;
                      end
                      chg_nstep=0;
                      switch(n_step)   %   number of eBDF3 steps
                         case(1)   %   single step
                            dt0=dt(n); dt1=dt(n-1); dt2=dt(n-2); n_repet=1; n_st=1;
                         case(2)   %   two step
                            switch(stab)   %  stable/unstable variant
                                case{1,3}   %   absolute
                                  dt1=dt(n-1); dt2=dt(n-2); n_repet=1; n_st=2;
                                  dt_crit=dt1*(dt1+dt2)/(2*dt1+dt2);
                                  tol=1.e-3*eps_t;
                                  dt_low=0; dt_hi=dt_crit; 
                                  while(abs(dt_low-dt_hi)>tol)
                                     dt_curr=0.5*(dt_low+dt_hi);
                                     c2=-(2*dt_curr*dt1+dt_curr*dt2-dt1*dt1-dt1*dt2)/...
                                         (dt1*dt1*(dt1+dt2)^2)*...
                                         (dt_curr+dt1)*(dt_curr+dt1+dt2);
                                     c0=-(dt_curr*dt_curr*(dt_curr+dt1))/...
                                         (dt2*(dt1+dt2)^2);
                                     a2=-c2; a0=-c0;
                                     if(a2>-a0)
                                        dt_hi=dt_curr;
                                     else
                                        dt_low=dt_curr;
                                     end
                                  end
                                  if(dt(n)<=dt_curr)
                                     dt0=dt(n); n_st=1; chg_nstep=1;
                                  else
                                     dt0=dt_curr;
                                  end
                                  if(stab==3)
                                     dt0=min(0.5*dt(n),dt_curr); 
                                  end
                               case(2)   %   border
                                  dt1=dt(n-1); dt2=dt(n-2); n_repet=1; n_st=2;
                                  if(dt(n)<=dt1*(dt1+dt2)/(2*dt1+dt2))
                                     n_st=1; chg_nstep=1;
                                  end
                                  dt0=min(dt(n),dt1*(dt1+dt2)/(2*dt1+dt2));
                                  case(4)   %   half-step
                                     dt0=0.5*dt(n); dt1=dt(n-1); dt2=dt(n-2);
                                     n_st=2; n_repet=1;  
                            end
                         case(3)   %   one/two step 
                            mx_pred1=mx_pred; my_pred1=my_pred; mz_pred1=mz_pred;
                            mx_pred2=mx_pred; my_pred2=my_pred; mz_pred2=mz_pred;
                            n_repet=2; 
                            n_step=3;
                         end
                         for r=1:n_repet
                            fprintf('PREDICTOR %1i\n',r); 
                            if(n_step==3)
                               switch(r)
                                  case(1)   %   single step
                                     dt0=dt(n); dt1=dt(n-1); dt2=dt(n-2); n_st=1;
                                  case(2)   %   two step
                                     switch(stab)   %  stable/unstable variant
                                         case{1,3}   %   absolute
                                           dt1=dt(n-1); dt2=dt(n-2); n_st=2;
                                           dt_crit=dt1*(dt1+dt2)/(2*dt1+dt2);
                                           tol=1.e-3*eps_t;
                                           dt_low=0; dt_hi=dt_crit; 
                                           while(abs(dt_low-dt_hi)>tol)
                                              dt_curr=0.5*(dt_low+dt_hi);
                                              c2=-(2*dt_curr*dt1+dt_curr*dt2-dt1*dt1-dt1*dt2)/...
                                                  (dt1*dt1*(dt1+dt2)^2)*...
                                                  (dt_curr+dt1)*(dt_curr+dt1+dt2);
                                              c0=-(dt_curr*dt_curr*(dt_curr+dt1))/...
                                                  (dt2*(dt1+dt2)^2);
                                              a2=-c2; a0=-c0;
                                              if(a2>-a0)
                                                 dt_hi=dt_curr;
                                              else
                                                 dt_low=dt_curr;
                                              end
                                           end
                                           if(dt(n)<=dt_curr)
                                              dt0=dt(n); n_st=1; chg_nstep=1;
                                           else
                                              dt0=dt_curr;
                                           end
                                           if(stab==3)
                                              dt0=min(0.5*dt(n),dt_curr); 
                                           end
                                        case(2)   %   border
                                           dt1=dt(n-1); dt2=dt(n-2); n_st=2;
                                           if(dt(n)<=dt1*(dt1+dt2)/(2*dt1+dt2))
                                              n_st=1; chg_nstep=1;
                                           end
                                           dt0=min(dt(n),dt1*(dt1+dt2)/(2*dt1+dt2));
                                     end
                               end
                            end
                            for i=1:n_st
                               fprintf('   eBDF3 predictor step %2i, dt=%8.3f\n',i,dt0); 
                               b=dt0/(dt1*(dt1+dt2))*(dt0+dt1)*(dt0+dt1+dt2);
                               c2=-(2*dt0*dt1+dt0*dt2-dt1*dt1-dt1*dt2)/...
                                   (dt1*dt1*(dt1+dt2)^2)*(dt0+dt1)*(dt0+dt1+dt2);
                               c1=(dt0*dt0)/(dt1*dt1*dt2)*(dt0+dt1+dt2);
                               c0=-(dt0*dt0*(dt0+dt1))/(dt2*(dt1+dt2)^2);
                               fprintf('   Characteristic polynomial: a2=%6.3f, a1=%6.3f, a0=%6.3f\n',-c2,-c1,-c0);
                               fprintf('   a0+a1+a2=%6.3f\n',-c0-c1-c2);
                               fprintf(fid,'   Characteristic polynomial: a2=%6.3f, a1=%6.3f, a0=%6.3f\n',-c2,-c1,-c0);
                               fprintf(fid,'   a0+a1+a2=%6.3f\n',-c0-c1-c2);
                               xi=roots([1,-c2,-c1,-c0]);
                               fprintf('   xi(1)=%8.4f, xi(2)=%8.4f, xi(3)=%8.4f\n',xi(1),xi(2),xi(3));
                               fprintf('   Characteristic polynomial: b1=%6.3f, b0=%6.3f\n',1-c2,c0); 
                               fprintf(fid,'   xi(1)=%8.4f, xi(2)=%8.4f, xi(3)=%8.4f\n',xi(1),xi(2),xi(3));
                               fprintf(fid,'   Characteristic polynomial: b1=%6.3f, b0=%6.3f\n',1-c2,c0); 
                               zeta=roots([1,1-c2,c0]);
                               fprintf('   xi(2)=%8.4f, xi(3)=%8.4f\n',zeta(1),zeta(2));
                               fprintf(fid,'   xi(2)=%8.4f, xi(3)=%8.4f\n',zeta(1),zeta(2));
                               fprintf('-----------------------------------------------\n');
                               fprintf(fid,'-----------------------------------------------\n');
                               mx_ebdf3=b*(-1/(1+alpha^2)*(my_p0*hz_p0-mz_p0*hy_p0)-alpha/(1+alpha^2)*...
                                        (my_p0*mx_p0*hy_p0+mx_p0*mz_p0*hz_p0-(my_p0^2+mz_p0^2)*hx_p0))+...
                                        c2*mx_p0+c1*mx_p1+c0*mx_p2;
                               my_ebdf3=b*(-1/(1+alpha^2)*(mz_p0*hx_p0-mx_p0*hz_p0)-alpha/(1+alpha^2)*...
                                        (mx_p0*my_p0*hx_p0+my_p0*mz_p0*hz_p0-(mx_p0^2+mz_p0^2)*hy_p0))+...
                                        c2*my_p0+c1*my_p1+c0*my_p2;
                               mz_ebdf3=b*(-1/(1+alpha^2)*(mx_p0*hy_p0-my_p0*hx_p0)-alpha/(1+alpha^2)*...
                                        (mx_p0*mz_p0*hx_p0+my_p0*mz_p0*hy_p0-(mx_p0^2+my_p0^2)*hz_p0))+...
                                        c2*mz_p0+c1*mz_p1+c0*mz_p2;
                               if(i==n_st)   %   final predictor step
                                  switch(n_step)
                                      case{1,2}   %   one or two step  
                                        m_ebdf3=[mx_ebdf3;my_ebdf3;mz_ebdf3]; 
                                       % if(hist==2)
                                       %    mx_pred=[mx_pred,mx_ebdf3]; 
                                       %    my_pred=[my_pred,my_ebdf3]; 
                                       %    mz_pred=[mz_pred,mz_ebdf3];
                                       % end
                                     case(3)   %   one/two step
                                        switch(r)
                                           case(1)
                                               m_ebdf31=[mx_ebdf3;my_ebdf3;mz_ebdf3];
                                           case(2)
                                               m_ebdf32=[mx_ebdf3;my_ebdf3;mz_ebdf3];
                                        end
                                   %     if(hist==2)
                                   %        switch(r)
                                   %           case(1)
                                   %              mx_pred1=[mx_pred1,mx_ebdf3]; 
                                   %              my_pred1=[my_pred1,my_ebdf3]; 
                                   %              mz_pred1=[mz_pred1,mz_ebdf3];
                                   %           case(2)
                                   %              mx_pred2=[mx_pred2,mx_ebdf3];
                                   %              my_pred2=[my_pred2,my_ebdf3];
                                   %              mz_pred2=[mz_pred2,mz_ebdf3];   
                                   %        end
                                   %     end
                                  end
                                  if(n_step~=3)
                                     fprintf('   eBDF3:    mx(%8.4f)=%7.4f; my(%8.4f)=%7.4f; mz(%8.4f)=%7.4f\n',t_c,mx_ebdf3,t_c,my_ebdf3,t_c,mz_ebdf3);
                                     fprintf(fid,'   eBDF3:    mx(%8.4f)=%7.4f; my(%8.4f)=%7.4f; mz(%8.4f)=%7.4f\n',t_c,mx_ebdf3,t_c,my_ebdf3,t_c,mz_ebdf3); 
                                  else
                                     switch(r)
                                        case(1)
                                           fprintf('   eBDF3:    mx(%8.4f)=%7.4f; my(%8.4f)=%7.4f; mz(%8.4f)=%7.4f\n',t_c,m_ebdf31(1),t_c,m_ebdf31(2),t_c,m_ebdf31(3));
                                           fprintf(fid,'   eBDF3:    mx(%8.4f)=%7.4f; my(%8.4f)=%7.4f; mz(%8.4f)=%7.4f\n',t_c,m_ebdf31(1),t_c,m_ebdf31(2),t_c,m_ebdf31(3)); 
                                        case(2)
                                           fprintf('   eBDF3:    mx(%8.4f)=%7.4f; my(%8.4f)=%7.4f; mz(%8.4f)=%7.4f\n',t_c,m_ebdf32(1),t_c,m_ebdf32(2),t_c,m_ebdf32(3));
                                           fprintf(fid,'   eBDF3:    mx(%8.4f)=%7.4f; my(%8.4f)=%7.4f; mz(%8.4f)=%7.4f\n',t_c,m_ebdf32(1),t_c,m_ebdf32(2),t_c,m_ebdf32(3)); 
                                     end
                                  end
                               else   %  first step in two step method
                                  dt2=dt1; dt1=dt0; dt0=dt(n)-dt0;
                                  mx_p2=mx_p1; my_p2=my_p1; mz_p2=mz_p1;
                                  mx_p1=mx_p0; my_p1=my_p0; mz_p1=mz_p0;
                                  mx_p0=mx_ebdf3; my_p0=my_ebdf3; mz_p0=mz_ebdf3;
                                  if(field==2)
                                     me_0=mx_p0*ex+my_p0*ey+mz_p0*ez;
                                     me_1=mx_p1*ex+my_p1*ey+mz_p1*ez;
                                     me_2=mx_p2*ex+my_p2*ey+mz_p2*ez;
                                     hx_f=hx(n); hy_f=hy(n); hz_f=hz(n);   %%%%% hoolds only for constant h_ext
                                     hx_p0=hx_f+k1*me_0*ex; hy_p0=hy_f+k1*me_0*ey; hz_p0=hz_f+k1*me_0*ez;   %%%%
                                     hx_p1=hx(n)+k1*me_1*ex; hy_p1=hy(n)+k1*me_1*ey; hz_p1=hz(n)+k1*me_1*ez;
                                     hx_p2=hx(n-1)+k1*me_2*ex; hy_p2=hy(n-1)+k1*me_2*ey; hz_p2=hz(n-1)+k1*me_2*ez; 
                                  end
                               end
                            end
                         end
                         if(chg_nstep==1)
                            n_st=2; chg_nstep=0; 
                         end
                      case(2)   %   AB3
                         mx_p0=mx(n); my_p0=my(n); mz_p0=mz(n);
                         mx_p1=mx(n-1); my_p1=my(n-1); mz_p1=mz(n-1);
                         mx_p2=mx(n-2); my_p2=my(n-2); mz_p2=mz(n-2);
                         switch(field)
                            case(1)
                               hx_p0=hx(n); hy_p0=hy(n); hz_p0=hz(n);
                               hx_p1=hx(n-1); hy_p1=hy(n-1); hz_p1=hz(n-1);
                               hx_p2=hx(n-2); hy_p2=hy(n-2); hz_p2=hz(n-2);
                            case(2)
                               me_0=mx(n)*ex+my(n)*ey+mz(n)*ez;
                               me_1=mx(n-1)*ex+my(n-1)*ey+mz(n-1)*ez;
                               me_2=mx(n-2)*ex+my(n-2)*ey+mz(n-2)*ez;
                               hx_p0=hx(n)+k1*me_0*ex; hy_p0=hy(n)+k1*me_0*ey; hz_p0=hz(n)+k1*me_0*ez;
                               hx_p1=hx(n-1)+k1*me_1*ex; hy_p1=hy(n-1)+k1*me_1*ey; hz_p1=hz(n-1)+k1*me_1*ez;
                               hx_p2=hx(n-2)+k1*me_2*ex; hy_p2=hy(n-2)+k1*me_2*ey; hz_p2=hz(n-2)+k1*me_2*ez; 
                         end
                         dt0=dt(n); dt1=dt(n-1); dt2=dt(n-2);
                         mx_ab3=mx_p0+(dt0*dt0*(dt1+2/3*dt0))/(2*dt2*(dt2+dt1))*...
                                      (-1/(1+alpha^2)*(my_p2*hz_p2-mz_p2*hy_p2)-alpha/(1+alpha^2)*...
                                      (my_p2*mx_p2*hy_p2+mx_p2*mz_p2*hz_p2-(my_p2^2+mz_p2^2)*hx_p2))-...
                                      (dt0*dt0*(dt2+dt1+2/3*dt0))/(2*dt2*dt1)*...
                                      (-1/(1+alpha^2)*(my_p1*hz_p1-mz_p1*hy_p1)-alpha/(1+alpha^2)*...
                                      (my_p1*mx_p1*hy_p1+mx_p1*mz_p1*hz_p1-(my_p1^2+mz_p1^2)*hx_p1))+...
                                      (dt1*dt0*(2*dt2+2*dt1+dt0)+dt0*dt0*(dt2+dt1+2/3*dt0))/(2*dt1*(dt2+dt1))*...
                                      (-1/(1+alpha^2)*(my_p0*hz_p0-mz_p0*hy_p0)-alpha/(1+alpha^2)*...
                                      (my_p0*mx_p0*hy_p0+mx_p0*mz_p0*hz_p0-(my_p0^2+mz_p0^2)*hx_p0));
                         my_ab3=my_p0+(dt0*dt0*(dt1+2/3*dt0))/(2*dt2*(dt2+dt1))*...
                                      (-1/(1+alpha^2)*(mz_p2*hx_p2-mx_p2*hz_p2)-alpha/(1+alpha^2)*...
                                      (mx_p2*my_p2*hx_p2+my_p2*mz_p2*hz_p2-(mx_p2^2+mz_p2^2)*hy_p2))-...
                                      (dt0*dt0*(dt2+dt1+2/3*dt0))/(2*dt2*dt1)*...
                                      (-1/(1+alpha^2)*(mz_p1*hx_p1-mx_p1*hz_p1)-alpha/(1+alpha^2)*...
                                      (mx_p1*my_p1*hx_p1+my_p1*mz_p1*hz_p1-(mx_p1^2+mz_p1^2)*hy_p1))+...
                                      (dt1*dt0*(2*dt2+2*dt1+dt0)+dt0*dt0*(dt2+dt1+2/3*dt0))/(2*dt1*(dt2+dt1))*...
                                      (-1/(1+alpha^2)*(mz_p0*hx_p0-mx_p0*hz_p0)-alpha/(1+alpha^2)*...
                                      (mx_p0*my_p0*hx_p0+my_p0*mz_p0*hz_p0-(mx_p0^2+mz_p0^2)*hy_p0));
                         mz_ab3=mz_p0+(dt0*dt0*(dt1+2/3*dt0))/(2*dt2*(dt2+dt1))*...
                                      (-1/(1+alpha^2)*(mx_p2*hy_p2-my_p2*hx_p2)-alpha/(1+alpha^2)*...
                                      (mx_p2*mz_p2*hx_p2+my_p2*mz_p2*hy_p2-(mx_p2^2+my_p2^2)*hz_p2))-...
                                      (dt0*dt0*(dt2+dt1+2/3*dt0))/(2*dt2*dt1)*...
                                      (-1/(1+alpha^2)*(mx_p1*hy_p1-my_p1*hx_p1)-alpha/(1+alpha^2)*...
                                      (mx_p1*mz_p1*hx_p1+my_p1*mz_p1*hy_p1-(mx_p1^2+my_p1^2)*hz_p1))+...
                                      (dt1*dt0*(2*dt2+2*dt1+dt0)+dt0*dt0*(dt2+dt1+2/3*dt0))/(2*dt1*(dt2+dt1))*...
                                      (-1/(1+alpha^2)*(mx_p0*hy_p0-my_p0*hx_p0)-alpha/(1+alpha^2)*...
                                      (mx_p0*mz_p0*hx_p0+my_p0*mz_p0*hy_p0-(mx_p0^2+my_p0^2)*hz_p0));        
                         m_ab3=[mx_ab3;my_ab3;mz_ab3];
                      case(3)   %   RK3
                         switch(field)
                            case(1)
                               kx1=-1/(1+alpha^2)*(my(n)*hz(n)-mz(n)*hy(n))-alpha/(1+alpha^2)*...
                                   (my(n)*mx(n)*hy(n)+mx(n)*mz(n)*hz(n)-(my(n)^2+mz(n)^2)*hx(n));
                               ky1=-1/(1+alpha^2)*(mz(n)*hx(n)-mx(n)*hz(n))-alpha/(1+alpha^2)*...
                                   (mx(n)*my(n)*hx(n)+my(n)*mz(n)*hz(n)-(mx(n)^2+mz(n)^2)*hy(n));
                               kz1=-1/(1+alpha^2)*(mx(n)*hy(n)-my(n)*hx(n))-alpha/(1+alpha^2)*...
                                   (mx(n)*mz(n)*hx(n)+my(n)*mz(n)*hy(n)-(mx(n)^2+my(n)^2)*hz(n));
                               mx2=mx(n)+0.5*dt(n)*kx1; my2=my(n)+0.5*dt(n)*ky1; mz2=mz(n)+0.5*dt(n)*kz1;
                               kx2=-1/(1+alpha^2)*(my2*hz(n)-mz2*hy(n))-alpha/(1+alpha^2)*...
                                   (my2*mx2*hy(n)+mx2*mz2*hz(n)-(my2^2+mz2^2)*hx(n));
                               ky2=-1/(1+alpha^2)*(mz2*hx(n)-mx2*hz(n))-alpha/(1+alpha^2)*...
                                   (mx2*my2*hx(n)+my2*mz2*hz(n)-(mx2^2+mz2^2)*hy(n));
                               kz2=-1/(1+alpha^2)*(mx2*hy(n)-my2*hx(n))-alpha/(1+alpha^2)*...
                                   (mx2*mz2*hx(n)+my2*mz2*hy(n)-(mx2^2+my2^2)*hz(n));
                               mx3=mx(n)+dt(n)*(-kx1+2*kx2);
                               my3=my(n)+dt(n)*(-ky1+2*ky2);
                               mz3=mz(n)+dt(n)*(-kz1+2*kz2);
                               kx3=-1/(1+alpha^2)*(my3*hz(n)-mz3*hy(n))-alpha/(1+alpha^2)*...
                                   (my3*mx3*hy(n)+mx3*mz3*hz(n)-(my3^2+mz3^2)*hx(n));
                               ky3=-1/(1+alpha^2)*(mz3*hx(n)-mx3*hz(n))-alpha/(1+alpha^2)*...
                                  (mx3*my3*hx(n)+my3*mz3*hz(n)-(mx3^2+mz3^2)*hy(n));
                               kz3=-1/(1+alpha^2)*(mx3*hy(n)-my3*hx(n))-alpha/(1+alpha^2)*...
                                  (mx3*mz3*hx(n)+my3*mz3*hy(n)-(mx3^2+my3^2)*hz(n));
                            case(2)
                               me1=mx(n)*ex+my(n)*ey+mz(n)*ez;
                               hhx_1=hx(n)+k1*me1*ex; hhy_1=hy(n)+k1*me1*ey; hhz_1=hz(n)+k1*me1*ez;  
                               kx1=-1/(1+alpha^2)*(my(n)*hhz_1-mz(n)*hhy_1)-alpha/(1+alpha^2)*...
                                   (my(n)*mx(n)*hhy_1+mx(n)*mz(n)*hhz_1-(my(n)^2+mz(n)^2)*hhx_1);
                               ky1=-1/(1+alpha^2)*(mz(n)*hhx_1-mx(n)*hhz_1)-alpha/(1+alpha^2)*...
                                   (mx(n)*my(n)*hhx_1+my(n)*mz(n)*hhz_1-(mx(n)^2+mz(n)^2)*hhy_1);
                               kz1=-1/(1+alpha^2)*(mx(n)*hhy_1-my(n)*hhx_1)-alpha/(1+alpha^2)*...
                                   (mx(n)*mz(n)*hhx_1+my(n)*mz(n)*hhy_1-(mx(n)^2+my(n)^2)*hhz_1);
                               mx2=mx(n)+0.5*dt(n)*kx1; my2=my(n)+0.5*dt(n)*ky1; mz2=mz(n)+0.5*dt(n)*kz1;
                               me2=mx2*ex+my2*ey+mz2*ez;
                               hhx_2=hx(n)+k1*me2*ex; hhy_2=hy(n)+k1*me2*ey; hhz_2=hz(n)+k1*me2*ez; 
                               kx2=-1/(1+alpha^2)*(my2*hhz_2-mz2*hhy_2)-alpha/(1+alpha^2)*...
                                   (my2*mx2*hhy_2+mx2*mz2*hhz_2-(my2^2+mz2^2)*hhx_2);
                               ky2=-1/(1+alpha^2)*(mz2*hhx_2-mx2*hhz_2)-alpha/(1+alpha^2)*...
                                   (mx2*my2*hhx_2+my2*mz2*hhz_2-(mx2^2+mz2^2)*hhy_2);
                               kz2=-1/(1+alpha^2)*(mx2*hhy_2-my2*hhx_2)-alpha/(1+alpha^2)*...
                                   (mx2*mz2*hhx_2+my2*mz2*hhy_2-(mx2^2+my2^2)*hhz_2);
                               mx3=mx(n)+dt(n)*(-kx1+2*kx2);
                               my3=my(n)+dt(n)*(-ky1+2*ky2);
                               mz3=mz(n)+dt(n)*(-kz1+2*kz2);
                               me3=mx3*ex+my3*ey+mz3*ez;
                               hhx_3=hx(n)+k1*me3*ex; hhy_3=hy(n)+k1*me3*ey; hhz_3=hz(n)+k1*me3*ez;
                               kx3=-1/(1+alpha^2)*(my3*hhz_3-mz3*hhy_3)-alpha/(1+alpha^2)*...
                                   (my3*mx3*hhy_3+mx3*mz3*hhz_3-(my3^2+mz3^2)*hhx_3);
                               ky3=-1/(1+alpha^2)*(mz3*hhx_3-mx3*hhz_3)-alpha/(1+alpha^2)*...
                                   (mx3*my3*hhx_3+my3*mz3*hhz_3-(mx3^2+mz3^2)*hhy_3);
                               kz3=-1/(1+alpha^2)*(mx3*hhy_3-my3*hhx_3)-alpha/(1+alpha^2)*...
                                   (mx3*mz3*hhx_3+my3*mz3*hhy_3-(mx3^2+my3^2)*hhz_3);
                         end
                         mx_rk3=mx(n)+dt(n)/6*(kx1+4*kx2+kx3);
                         my_rk3=my(n)+dt(n)/6*(ky1+4*ky2+ky3);
                         mz_rk3=mz(n)+dt(n)/6*(kz1+4*kz2+kz3);
                         m_rk3=[mx_rk3;my_rk3;mz_rk3]; 
                     case(4)   %   AB2
                        mx_p0=mx(n); my_p0=my(n); mz_p0=mz(n);
                        mx_p1=mx(n-1); my_p1=my(n-1); mz_p1=mz(n-1);
                        switch(field)
                           case(1)
                              hx_p0=hx(n); hy_p0=hy(n); hz_p0=hz(n);
                              hx_p1=hx(n-1); hy_p1=hy(n-1); hz_p1=hz(n-1);
                           case(2)
                              me_0=mx(n)*ex+my(n)*ey+mz(n)*ez;
                              me_1=mx(n-1)*ex+my(n-1)*ey+mz(n-1)*ez;
                              hx_p0=hx(n)+k1*me_0*ex; hy_p0=hy(n)+k1*me_0*ey; hz_p0=hz(n)+k1*me_0*ez;
                              hx_p1=hx(n-1)+k1*me_1*ex; hy_p1=hy(n-1)+k1*me_1*ey; hz_p1=hz(n-1)+k1*me_1*ez;
                        end
                        dt0=dt(n); dt1=dt(n-1); 
                        mx_ab2=mx_p0+(dt0+dt0^2/(2*dt1))*(-1/(1+alpha^2)*(my_p0*hz_p0-mz_p0*hy_p0)-...
                                alpha/(1+alpha^2)*(my_p0*mx_p0*hy_p0+mx_p0*mz_p0*hz_p0-(my_p0^2+mz_p0^2)*hx_p0))-...
                                dt0^2/(2*dt1)*(-1/(1+alpha^2)*(my_p1*hz_p1-mz_p1*hy_p1)-...
                                alpha/(1+alpha^2)*(my_p1*mx_p1*hy_p1+mx_p1*mz_p1*hz_p1-(my_p1^2+mz_p1^2)*hx_p1));
                        my_ab2=my_p0+(dt0+dt0^2/(2*dt1))*(-1/(1+alpha^2)*(mz_p0*hx_p0-mx_p0*hz_p0)-...
                                alpha/(1+alpha^2)*(mx_p0*my_p0*hx_p0+my_p0*mz_p0*hz_p0-(mx_p0^2+mz_p0^2)*hy_p0))-...
                                dt0^2/(2*dt1)*(-1/(1+alpha^2)*(mz_p1*hx_p1-mx_p1*hz_p1)-...
                                alpha/(1+alpha^2)*(mx_p1*my_p1*hx_p1+my_p1*mz_p1*hz_p1-(mx_p1^2+mz_p1^2)*hy_p1));
                        mz_ab2=mz_p0+(dt0+dt0^2/(2*dt1))*(-1/(1+alpha^2)*(mx_p0*hy_p0-my_p0*hx_p0)-...
                                alpha/(1+alpha^2)*(mx_p0*mz_p0*hx_p0+my_p0*mz_p0*hy_p0-(mx_p0^2+my_p0^2)*hz_p0))-...
                                dt0^2/(2*dt1)*(-1/(1+alpha^2)*(mx_p1*hy_p1-my_p1*hx_p1)-...
                                alpha/(1+alpha^2)*(mx_p1*mz_p1*hx_p1+my_p1*mz_p1*hy_p1-(mx_p1^2+my_p1^2)*hz_p1));
                        m_ab2=[mx_ab2;my_ab2;mz_ab2];   
                  end
              %   IMR corrector step
                  mx_n=mx(n); my_n=my(n); mz_n=mz(n);   %   Newton init. guess
                  m_n=[mx_n;my_n;mz_n];
                  hx_n=0; hy_n=0; hz_n=-1.1;   %  external field
                  hx=[hx,hx_n]; hy=[hy,hy_n]; hz=[hz,hz_n];
                  mx_h=(mx_n+mx(n))/2; my_h=(my_n+my(n))/2; mz_h=(mz_n+mz(n))/2;
                  hx_h=(hx(n+1)+hx(n))/2; hy_h=(hy(n+1)+hy(n))/2; hz_h=(hz(n+1)+hz(n))/2;
                  switch(field)
                     case(1)
                        rx=mx_n+dtc*(1/(1+alpha^2)*(my_h*hz_h-mz_h*hy_h)+alpha/(1+alpha^2)*...
                                    (my_h*mx_h*hy_h+mx_h*mz_h*hz_h-(my_h^2+mz_h^2)*hx_h))-mx(n);
                        ry=my_n+dtc*(1/(1+alpha^2)*(mz_h*hx_h-mx_h*hz_h)+alpha/(1+alpha^2)*...
                                    (mx_h*my_h*hx_h+my_h*mz_h*hz_h-(mx_h^2+mz_h^2)*hy_h))-my(n);
                        rz=mz_n+dtc*(1/(1+alpha^2)*(mx_h*hy_h-my_h*hx_h)+alpha/(1+alpha^2)*...
                                    (mx_h*mz_h*hx_h+my_h*mz_h*hy_h-(mx_h^2+my_h^2)*hz_h))-mz(n); 
                     case(2)
                         me_h=(mx_n+mx(n))/2*ex+(my_n+my(n))/2*ey+(mz_n+mz(n))/2*ez;
                         hhx_h=hx_h+k1*me_h*ex; hhy_h=hy_h+k1*me_h*ey; hhz_h=hz_h+k1*me_h*ez;
                         rx=mx_n+dtc*(1/(1+alpha^2)*(my_h*hhz_h-mz_h*hhy_h)+alpha/(1+alpha^2)*...
                                     (my_h*mx_h*hhy_h+mx_h*mz_h*hhz_h-(my_h^2+mz_h^2)*hhx_h))-mx(n);
                         ry=my_n+dtc*(1/(1+alpha^2)*(mz_h*hhx_h-mx_h*hhz_h)+alpha/(1+alpha^2)*...
                                     (mx_h*my_h*hhx_h+my_h*mz_h*hhz_h-(mx_h^2+mz_h^2)*hhy_h))-my(n);
                         rz=mz_n+dtc*(1/(1+alpha^2)*(mx_h*hhy_h-my_h*hhx_h)+alpha/(1+alpha^2)*...
                                     (mx_h*mz_h*hhx_h+my_h*mz_h*hhy_h-(mx_h^2+my_h^2)*hhz_h))-mz(n); 
                  end
                  r_vec=[rx;ry;rz]; r0_norm=norm(r_vec,2); n_new=0; r_norm=1;
                  fprintf('   ||r_nl(0)||_2=%10.4e\n',r0_norm);
                  while(r_norm>=max(1.e-9*r0_norm,1.2e-15))
                     switch(field)
                        case(1)     
                           j11=1+(alpha*dtc)/(2*(1+alpha^2))*(my_h*hy_h+mz_h*hz_h);
                           j12=dtc/(2*(1+alpha^2))*hz_h+(alpha*dtc)/(2*(1+alpha^2))*(mx_h*hy_h-2*my_h*hx_h);
                           j13=-dtc/(2*(1+alpha^2))*hy_h+(alpha*dtc)/(2*(1+alpha^2))*(mx_h*hz_h-2*mz_h*hx_h);
                           j21=-dtc/(2*(1+alpha^2))*hz_h+(alpha*dtc)/(2*(1+alpha^2))*(my_h*hx_h-2*mx_h*hy_h);
                           j22=1+(alpha*dtc)/(2*(1+alpha^2))*(mx_h*hx_h+mz_h*hz_h);
                           j23=dtc/(2*(1+alpha^2))*hx_h+(alpha*dtc)/(2*(1+alpha^2))*(my_h*hz_h-2*mz_h*hy_h);
                           j31=dtc/(2*(1+alpha^2))*hy_h+(alpha*dtc)/(2*(1+alpha^2))*(mz_h*hx_h-2*mx_h*hz_h);
                           j32=-dtc/(2*(1+alpha^2))*hx_h+(alpha*dtc)/(2*(1+alpha^2))*(mz_h*hy_h-2*my_h*hz_h);
                           j33=1+(alpha*dtc)/(2*(1+alpha^2))*(mx_h*hx_h+my_h*hy_h);
                        case(2)
                           j11=1+(dtc/(2*(1+alpha^2)))*(k1*(my_h*ex*ez-mz_h*ex*ey))+(dtc*alpha)/(2*(1+alpha^2))*...
                                 (my_h*hhy_h+k1*my_h*mx_h*ex*ey+mz_h*hhz_h+k1*mx_h*mz_h*ex*ez-k1*(my_h^2+mz_h^2)*ex*ex);
                           j12=dtc/(2*(1+alpha^2))*(hhz_h-k1*mz_h*ey*ey+k1*my_h*ey*ez)+(dtc*alpha)/(2*(1+alpha^2))*...
                                 (mx_h*hy_h+k1*mx_h*my_h*ey*ey+k1*mx_h*mz_h*ey*ez-2*my_h*hhx_h-k1*(my_h^2+mz_h^2)*ey*ex);
                           j13=dtc/(2*(1+alpha^2))*(k1*my_h*ez*ez-hhy_h-k1*mz_h*ez*ey)+(dtc*alpha)/(2*(1+alpha^2))*...
                                 (k1*my_h*mx_h*ez*ey+mx_h*hhz_h+k1*mx_h*mz_h*ez*ez-2*mz_h*hhx_h-k1*(my_h^2+mz_h^2)*ez*ex);
                           j21=dtc/(2*(1+alpha^2))*(k1*mz_h*ex*ex-hhz_h-k1*mx_h*ex*ez)+(dtc*alpha)/(2*(1+alpha^2))*...
                                 (my_h*hhx_h+k1*mx_h*my_h*ex*ex+k1*my_h*mz_h*ex*ez-2*mx_h*hhy_h-k1*(mx_h^2+mz_h^2)*ex*ey);
                           j22=1+(dtc/(2*(1+alpha^2)))*(k1*(mz_h*ey*ex-mx_h*ey*ez))+(dtc*alpha)/(2*(1+alpha^2))*...
                                 (mx_h*hx_h+k1*mx_h*my_h*ey*ex+mz_h*hhz_h+k1*my_h*mz_h*ey*ez-k1*(mx_h^2+mz_h^2)*ey*ey);
                           j23=dtc/(2*(1+alpha^2))*(hhx_h+k1*mz_h*ez*ex-k1*mx_h*ez*ez)+(dtc*alpha)/(2*(1+alpha^2))*...
                                 (k1*mx_h*my_h*ez*ex+my_h*hhz_h+k1*my_h*mz_h*ez*ez-2*mz_h*hhy_h-k1*(mx_h^2+mz_h^2)*ez*ey);
                           j31=dtc/(2*(1+alpha^2))*(hhy_h-k1*my_h*ex*ex+k1*mx_h*ex*ey)+(dtc*alpha)/(2*(1+alpha^2))*...
                                 (mz_h*hhx_h+k1*mx_h*mz_h*ex*ex+k1*my_h*mz_h*ex*ey-2*mx_h*hhz_h-k1*(mx_h^2+my_h^2)*ex*ez);
                           j32=dtc/(2*(1+alpha^2))*(k1*mx_h*ey*ey-hhx_h-k1*my_h*ey*ex)+(dtc*alpha)/(2*(1+alpha^2))*...
                                 (k1*mx_h*mz_h*ey*ex+mz_h*hhy_h+k1*my_h*mz_h*ey*ey-2*my_h*hhz_h-k1*(mx_h^2+my_h^2)*ey*ez);
                           j33=1+(dtc/(2*(1+alpha^2)))*(k1*(mx_h*ez*ey-my_h*ez*ex))+(dtc*alpha)/(2*(1+alpha^2))*...
                                 (mx_h*hhx_h+k1*mx_h*mz_h*ez*ex+my_h*hhy_h+k1*my_h*mz_h*ez*ey-k1*(mx_h^2+my_h)^2*ez*ez);  
                     end
                     J=[j11,j12,j13;j21,j22,j23;j31,j32,j33];
                     dm_n=J\r_vec;
                     m_n=m_n-dm_n;
                     mx_n=m_n(1); my_n=m_n(2); mz_n=m_n(3);
                     mx_h=(mx_n+mx(n))/2; my_h=(my_n+my(n))/2; mz_h=(mz_n+mz(n))/2;
                     switch(field)
                        case(1)
                           rx=mx_n+dtc*(1/(1+alpha^2)*(my_h*hz_h-mz_h*hy_h)+alpha/(1+alpha^2)*...
                                       (my_h*mx_h*hy_h+mx_h*mz_h*hz_h-(my_h^2+mz_h^2)*hx_h))-mx(n);
                           ry=my_n+dtc*(1/(1+alpha^2)*(mz_h*hx_h-mx_h*hz_h)+alpha/(1+alpha^2)*...
                                       (mx_h*my_h*hx_h+my_h*mz_h*hz_h-(mx_h^2+mz_h^2)*hy_h))-my(n);
                           rz=mz_n+dtc*(1/(1+alpha^2)*(mx_h*hy_h-my_h*hx_h)+alpha/(1+alpha^2)*...
                                       (mx_h*mz_h*hx_h+my_h*mz_h*hy_h-(mx_h^2+my_h^2)*hz_h))-mz(n);
                        case(2)
                           me_h=(mx_n+mx(n))/2*ex+(my_n+my(n))/2*ey+(mz_n+mz(n))/2*ez;
                           hhx_h=hx_h+k1*me_h*ex; hhy_h=hy_h+k1*me_h*ey; hhz_h=hz_h+k1*me_h*ez;
                           rx=mx_n+dtc*(1/(1+alpha^2)*(my_h*hhz_h-mz_h*hhy_h)+alpha/(1+alpha^2)*...
                                       (my_h*mx_h*hhy_h+mx_h*mz_h*hhz_h-(my_h^2+mz_h^2)*hhx_h))-mx(n);
                           ry=my_n+dtc*(1/(1+alpha^2)*(mz_h*hhx_h-mx_h*hhz_h)+alpha/(1+alpha^2)*...
                                       (mx_h*my_h*hhx_h+my_h*mz_h*hhz_h-(mx_h^2+mz_h^2)*hhy_h))-my(n);
                           rz=mz_n+dtc*(1/(1+alpha^2)*(mx_h*hhy_h-my_h*hhx_h)+alpha/(1+alpha^2)*...
                                       (mx_h*mz_h*hhx_h+my_h*mz_h*hhy_h-(mx_h^2+my_h^2)*hhz_h))-mz(n);    
                     end
                     r_vec=[rx;ry;rz]; r_norm=norm(r_vec,2);
                     n_new=n_new+1;
                  end
                  fprintf('   Newton method converged in %1i steps\n',n_new);
                  fprintf('   ||r_nl(%2i)||_2=%10.4e\n',n_new,r_norm); 
                  mx=[mx,mx_n]; my=[my,my_n]; mz=[mz,mz_n];
                  if(field==2)
                     me=mx_n*ex+my_n*ey+mz_n*ez;
                     hmcx=[hmcx,k1*me*ex]; hmcy=[hmcy,k1*me*ey]; hmcz=[hmcz,k1*me*ez];
                  end
                  m_imr=[mx_n;my_n;mz_n];
                  switch(predict)
                     case(1)   %   eBDF3
                        if(n_step~=3)
                           fprintf('   eBDF3: mx(%8.4f)=%10.4f; my(%8.4f)=%10.4f; mz(%8.4f)=%10.4f\n',t_c,m_ebdf3(1),t_c,m_ebdf3(2),t_c,m_ebdf3(3));
                           fprintf(fid,'   eBDF3: mx(%8.4f)=%10.4f; my(%8.4f)=%10.4f; mz(%8.4f)=%10.4f\n',t_c,m_ebdf3(1),t_c,m_ebdf3(2),t_c,m_ebdf3(3));
                        else
                           fprintf('   eBDF3-1: mx(%8.4f)=%10.4f; my(%8.4f)=%10.4f; mz(%8.4f)=%10.4f\n',t_c,m_ebdf31(1),t_c,m_ebdf31(2),t_c,m_ebdf31(3));
                           fprintf(fid,'   eBDF3-1: mx(%8.4f)=%10.4f; my(%8.4f)=%10.4f; mz(%8.4f)=%10.4f\n',t_c,m_ebdf31(1),t_c,m_ebdf31(2),t_c,m_ebdf31(3));
                           fprintf('   eBDF3-2: mx(%8.4f)=%10.4f; my(%8.4f)=%10.4f; mz(%8.4f)=%10.4f\n',t_c,m_ebdf32(1),t_c,m_ebdf32(2),t_c,m_ebdf32(3));
                           fprintf(fid,'   eBDF3-2: mx(%8.4f)=%10.4f; my(%8.4f)=%10.4f; mz(%8.4f)=%10.4f\n',t_c,m_ebdf32(1),t_c,m_ebdf32(2),t_c,m_ebdf32(3));
                        end
                     case(2)   %   AB3
                        fprintf('   AB3: mx(%8.4f)=%10.4f; my(%8.4f)=%10.4f; mz(%8.4f)=%10.4f\n',t_c,m_ab3(1),t_c,m_ab3(2),t_c,m_ab3(3));
                        fprintf(fid,'   AB3: mx(%8.4f)=%10.4f; my(%8.4f)=%10.4f; mz(%8.4f)=%10.4f\n',t_c,m_ab3(1),t_c,m_ab3(2),t_c,m_ab3(3));
                     case(3)   %   RK3
                        fprintf('   RK3: mx(%8.4f)=%10.4f; my(%8.4f)=%10.4f; mz(%8.4f)=%10.4f\n',t_c,m_rk3(1),t_c,m_rk3(2),t_c,m_rk3(3));  
                        fprintf(fid,'   RK3: mx(%8.4f)=%10.4f; my(%8.4f)=%10.4f; mz(%8.4f)=%10.4f\n',t_c,m_rk3(1),t_c,m_rk3(2),t_c,m_rk3(3)); 
                     case(4)   %   AB2
                        fprintf('   AB2: mx(%8.4f)=%10.4f; my(%8.4f)=%10.4f; mz(%8.4f)=%10.4f\n',t_c,m_ab2(1),t_c,m_ab2(2),t_c,m_ab2(3));  
                        fprintf(fid,'   AB2: mx(%8.4f)=%10.4f; my(%8.4f)=%10.4f; mz(%8.4f)=%10.4f\n',t_c,m_ab2(1),t_c,m_ab2(2),t_c,m_ab2(3));  
                  end
                  fprintf('   IMR:  mx(%8.4f)=%7.4f; my(%8.4f)=%7.4f; mz(%8.4f)=%7.4f\n',t_c,mx_n,t_c,my_n,t_c,mz_n);
                  fprintf(fid,'   IMR:  mx(%8.4f)=%7.4f; my(%8.4f)=%7.4f; mz(%8.4f)=%7.4f\n',t_c,mx_n,t_c,my_n,t_c,mz_n);
              %   LTE estimate
                  switch(predict)
                     case(1)   %   eBDF3 predictor
                        if(n_step~=3) 
                           d=m_imr-m_ebdf3; dh=[dh,norm(d,2)];
                        else
                           d1=m_imr-m_ebdf31; d2=m_imr-m_ebdf32;
                           d=min(norm(d1,2),norm(d2,2));
                           dh=[dh,min(norm(d1,2),norm(d2,2))];
                        end
                        if(n_step~=3)
                           fprintf('   ||d(%5i)||=%8.3e\n',n,norm(d,2));
                           fprintf(fid,'   ||d(%5i)||=%8.3e\n',n,norm(d,2));
                        else
                           fprintf('   ||d(%5i)||=%8.3e\n',n,min(norm(d1,2),norm(d2,2)));
                           fprintf(fid,'   ||d(%5i)||=%8.3e\n',n,min(norm(d1,2),norm(d2,2)));
                        end
                      case(2)   %   AB3 predictor
                         d=m_imr-m_ab3; dh=[dh,norm(d,2)];
                         fprintf('   ||d(%5i)||=%8.3e\n',n,norm(d,2));
                         fprintf(fid,'   ||d(%5i)||=%8.3e\n',n,norm(d,2));
                      case(3)   %   RK3 predictor
                         d=m_imr-m_rk3; dh=[dh,norm(d,2)];
                         fprintf('   ||d(%5i)||=%8.3e\n',n,norm(d,2));
                         fprintf(fid,'   ||d(%5i)||=%8.3e\n',n,norm(d,2));
                      case(4)   %   AB2 predictor
                         d=1/(3*(1+dt(n-1)/dt(n)))*(m_imr-m_ab2); dh=[dh,norm(d,2)];
                         fprintf('   ||d(%5i)||=%8.3e\n',n,norm(d,2));
                         fprintf(fid,'   ||d(%5i)||=%8.3e\n',n,norm(d,2));
                  end
                  if(fin==1) 
                     break;
                  end
                  dt_new=(eps_t/norm(d,2))^(1/3)*dt(n);  
                  if(dt_new<1.e-10) 
                     fprintf('   Aborting the time integrator...\n'); 
                     return;
                  end
                  if(t_c+dt_new>=t_end)
                     dt_new=abs(t_end-t_c);
                     t_c=t_end; fin=1;
                  else
                     t_c=t_c+dt_new;
                  end
                  dt=[dt,dt_new]; th=[th,t_c]; dtc=dt_new;
             end
             set(gcf,'Position',[575,240,700,700]);
             subplot(221); plot(th,mx,'r.-','MarkerSize',4); 
             axis([0 t_end -1.01 1.01]);
             title('x-component of the magnetization m');
             xlabel('t'); ylabel('mx');
             subplot(222); plot(th,my,'b.-','MarkerSize',4);
             axis([0 t_end -1.01 1.01]);
             title('y-component of the magnetization m');
             xlabel('t'); ylabel('my');
             subplot(223); plot(th,mz,'g.-','MarkerSize',4);
             axis([0 t_end -1.01 1.01]);
             title('z-component of the magnetization m');
             xlabel('t'); ylabel('mz');
             subplot(224); plot(th,[0,dt],'k.-','MarkerSize',4);
             axis([0 t_end 0 1.1*max(dt)]);
             title('Time step history');
             xlabel('t'); ylabel('dt'); 
          otherwise
              fprintf('Wrong integration mode\n');  
       end
    otherwise
        fprintf('Wrong scheme\n');
end
aviobj=VideoWriter('LLG.avi');
open(aviobj);
fig=figure(2);
set(gcf,'Position',[110,490,450,450]);
nframe=size(th);
step=nframe(2)/1000;
if(step<1)
   step=1;
else
   step=floor(step);
end   
for frame=1:step:nframe(2)
   quiver3(0,0,0,hx(1),hy(1),hz(1),'Color','r','LineWidth',1.5);
   hold on;
   if(field==2)
      quiver3(0,0,0,ex,ey,ez,'Color','g','LineWidth',1.5);
      hold on;
   end
   quiver3(0,0,0,mx(frame),my(frame),mz(frame),'Color','b','LineWidth',1.5);
   axis([-1.1*max(max(abs(mx)),1) 1.1*max(max(abs(mx)),1) ...
         -1.1*max(max(abs(my)),1) 1.1*max(max(abs(my)),1) ...
         -1.1*max(max(abs(mz)),1) 1.1*max(max(abs(mz)),1)]);
   xlabel('x'); ylabel('y'); zlabel('z');
   title(['Magnetization vector at t=',num2str(th(frame))]);
   f=getframe(fig);
   writeVideo(aviobj,f);
   hold off;
end
%close(fig);
close(aviobj);
close(2);
nn=size(th);
Et(1:nn(2))=0;
for i=1:nn(2)
   Et(i)=Et(i)-mx(i)*hx(i)-my(i)*hy(i)-mz(i)*hz(i);
   if(field==2)
       Et(i)=Et(i)-mx(i)*hmcx(i)-my(i)*hmcy(i)-mz(i)*hmcz(i);
   end
end
figure(3);
set(gcf,'Position',[260,40,300,900]);
subplot(311); plot3(mx,my,mz,'Color','red');
axis([-1.1*max(max(abs(mx)),1) 1.1*max(max(abs(mx)),1) ...
         -1.1*max(max(abs(my)),1) 1.1*max(max(abs(my)),1) ...
         -1.1*max(max(abs(mz)),1) 1.1*max(max(abs(mz)),1)]);
title('Magnetization vector trace');
xlabel('x'); ylabel('y'); zlabel('z');
m_mod=sqrt(mx.^2+my.^2+mz.^2);
subplot(312); plot(th,m_mod,'Color','magenta');
axis([0 max(th) 0.95*min(m_mod) 1.05*max(m_mod)]);
title('Modulus of the magnetization vector');
xlabel('t'); ylabel('|m|');
subplot(313); plot(th,Et,'Color','cyan');
axis([0 max(th) 1.1*min(Et) 1.1*max(Et)]);
title('Total energy of the system');
xlabel('t'); ylabel('E');
fprintf('---------------------------------------------------------\n')
fprintf('min(|m|)=%10.6f, max(|m|)=%10.6f\n',min(m_mod),max(m_mod));
fclose(fid);
return;
    
