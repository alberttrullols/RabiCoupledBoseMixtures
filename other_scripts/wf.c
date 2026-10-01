void ConstructGridPhononLuttingerAuto11(struct Grid *G) {
  DOUBLE k, kmin, kmax;
  DOUBLE y, ymin, ymax;
  DOUBLE precision = 1e-12;
  int search = ON;
  DOUBLE R, Rmin, Rmax;
  DOUBLE a;
  int iter = 1;

  a = aA;
  Message("\n  Jastrow term: ConstructGridPhononLuttingerAuto11 trial wavefunction\n");

  if(a>0) {
    Warning(" assuming repulsive case, with aA = %lf < 0\n", -a);
    a = -a;
  }

  AllocateWFGrid(G, 3);

  Message("  Kpar11= %"LE " \n", Kpar11);
  alpha11_1 = 1. / Kpar11;

  // search for Rpar
  R = Rmin = 1e-8;
  // (2.59) - k ^ 2 = alpha11_1 * (PI / L)*(PI / L)* ((alpha11_1 - 1.)*cotan(PI*R/L)*cotan(PI*R/L)-1.)
  k = kmin = sqrt(-alpha11_1 * (PI / L)*(PI / L)* ((alpha11_1 - 1.) / tan(PI*R / L) / tan(PI*R / L) - 1.));
  // (2.58) -k*tan(k*R + atan(1./(k*a))) = alpha11_1*PI/L*cotan(PI*R/L)
  ymin = -k * tan(k*R + atan(1. / (k*a))) - alpha11_1 * PI / L / tan(PI*R / L);

  R = Rmax = Lhalf;
  k = kmax = sqrt(-alpha11_1 * (PI / L)*(PI / L)* ((alpha11_1 - 1.) / tan(PI*R / L) / tan(PI*R / L) - 1.));
  ymax = -k * tan(k*R + atan(1. / (k*a))) - alpha11_1 * PI / L / tan(PI*R / L);

  if (fabs(ymax) < precision) {
    Message("  no iterations are needed\n");
    R = Rmax;
    search = OFF;
  }
  else {
    if (ymin*ymax > 0) Error("  cannot construct w.f. (11)");
  }

  // Solve k*tan(k*R + atan(1./(k*a))) = -PI*alpha11_1/L/tan(PI*R/L);
  // i.e. f'(R)/f(R)
  while (search) {
    R = (Rmin + Rmax) / 2.;
    k = sqrt(-alpha11_1 * (PI / L)*(PI / L)* ((alpha11_1 - 1.) / tan(PI*R / L) / tan(PI*R / L) - 1.));
    y = -k * tan(k*R + atan(1. / (k*a))) - alpha11_1 * PI / L / tan(PI*R / L);

    if (y*ymin < 0) {
      Rmax = R;
      ymax = y;
    }
    else {
      Rmin = R;
      ymin = y;
    }

    if (fabs(y) < precision) {
      R = (Rmax*ymin - Rmin * ymax) / (ymin - ymax);
      search = OFF;
    }

    if (iter++ == 1000) {
      Warning("  maximal number of iterations exceeded");
    }
  }
  k = sqrt(-alpha11_1 * (PI / L)*(PI / L)* ((alpha11_1 - 1.) / tan(PI*R / L) / tan(PI*R / L) - 1.));
  y = -k * tan(k*R + atan(1. / (k*a))) - alpha11_1 * PI / L / tan(PI*R / L);
  Message("done\n");

  Btrial11 = -Arctg(1 / (k*a)) / k;
  Atrial11 = Sin(PI*R / L);
  Atrial11 = pow(Sin(PI*R / L), alpha11_1) / Cos(k*(R - Btrial11));
  Rpar11 = R;
  sqE11 = k;
  sqE211 = sqE11 * sqE11;

  Message("    k = %"LE " , error %"LE "\n", k, y);
  Message("    R = %"LE " = %lf L/2\n", R, R / Lhalf);
  Message("    A = %"LE " \n", Atrial11);
  Message("    B = %"LE " \n", Btrial11);
  Message("    Equivalent Luttiner parameter K = 1/alpha = %"LE " \n", 1. / alpha11_1);
  //Message("    [Eloc(R-0) -Eloc(R+0)] / Eloc(R) = %"LE "\n", (InterpolateExactE11(G, R*0.99999) - InterpolateExactE11(G, R*1.00001)) / InterpolateExactE11(G, R));

  G->min = 0;
  G->max = L / 2;
  G->max2 = G->max*G->max;
}

// ideal bosons K -> \infty, alpha11_1 = 1/K = 0
// ideal fermions K = 1, alpha11_1 = 1/K = 1
DOUBLE InterpolateExactU11(DOUBLE x) {
#ifdef BC_ABSENT
  if(x>Lhalf) return 0;
#endif
  if(x<Rpar11)
    return Log(fabs(Atrial11*Cos(sqE11*(x - Btrial11))));
  else
    return Log(pow(Sin(PI*x / L), alpha11_1));
}

DOUBLE InterpolateExactFp11(DOUBLE x) {
#ifdef BC_ABSENT
  if(x>Lhalf) return 0;
#endif
  if(x<Rpar11)
    return -sqE11 * Tg(sqE11*(x - Btrial11));
  else
    return alpha11_1 * PI / (Lwf*Tg(PI*x / L));
}

DOUBLE InterpolateExactE11(DOUBLE x) {
  DOUBLE c;
#ifdef BC_ABSENT
  if(x>Lhalf) return 0;
#endif
  if(x<Rpar11) {
    c = Tg(sqE11*(x - Btrial11));
    return sqE211 * (1. + c*c);
  }
  else {
    c = 1 / Tg(PI*x / Lwf);
    return alpha11_1 * (PI*PI / (Lwf*Lwf))*(1 + c*c);
  }
}
#endif
