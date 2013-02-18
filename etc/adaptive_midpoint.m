#!/usr/local/bin/MathematicaScript -script

(* exact = Series[y[t + h], {h, 0, 3}] *)

(* f[y[t],t] = D[y, t] *)

(* midpoint = y[t] + h*f[( y[t + h] + y[t])/2 , t + h/2] *)

(* FullSimplify[midpoint - exact] *)

f = D[y[t], t]
midpointLTE := ymp - ynp1 == ((D[f,{t,2}] * f - 2 D[f,{t,1}]^2) * h^3)/24
AB2LTE := yab2 - ynp1 == (5/12) * h^3 * D[f,{t,2}]

eliminateDerivative3[correctorLTE_, predictorLTE_] :=
    Module[{ynp1Replacement, a},  (* Local variables must be explicitly declared here *)

           (* Get y''' without any dependence on exact solution by using
              the predictor solution *)
           ynp1Replacement := Solve[Eliminate[{correctorLTE, predictorLTE},
                                              ynp1],
                                    D[f,{t,2}]];

           (* Substitute in y''' and an approximation to y'' *)
           a = Replace[correctorLTE, ynp1Replacement, Infinity];
           (* b = Replace[a, approxDDy, Infinity]; *)

           Return[Simplify[a]]
    ]


CForm[eliminateDerivative3[midpointLTE, AB2LTE]]


(* Output of command: *)
(* Out[5]//CForm= List(ymp ==  *)
(*      ynp1 + (-6*(yab2 - ymp)*Derivative(1)(y)(t) +  *)
(*          5*Power(h,3)*Power(Derivative(2)(y)(t),2))/(6.*(-10 + Derivative(1)(y)(t)))) *)



(*
   In python: (truncation_error = ymp - ynp1)
   truncation_error = (-6.0 * (yab2 - ymp) * ydotn + 5.0 * h**3 * ydotdotn**2 ) \
                       / (6.0 * ( -10.0 + ydotn))
   *)


(* Check vs my working *)
(* Simplify[Solve[Eliminate[{midpointLTE, AB2LTE}, ynp1], D[f,{t,2}]]] *)
(* Simplify[((ymp - yab2)/h^3 + D[f,{t,1}]/12) / (f/24 - 5/12)] *)
(* Simplify[((1/2)*h^3* D[f,{t,1}]^2 *(-1/6 + f/(f/2 - 5)) + (1/2) * f * (ymp - yab2) / (f/2 -5))] *)
(* Simplify[-(1/12) h^3 D[f,{t,1}]^2 + ((1/2) f (ymp - yab2) + (1/2) f h^3 D[f,{t,1}]^2) / ((1/2) f - 5)] *)
(* Simplify[-(1/12) D[f,{t,1}]^2 h^3 + ((1/24) h^3 f (ymp - yab2) )/ (h^3 ((1/24) f - (5/12))) *)
(*          + ((1/24) f h^3 (1/12) D[f,{t,1}]^2 )/ ((1/24) f - 5/12)] *)
