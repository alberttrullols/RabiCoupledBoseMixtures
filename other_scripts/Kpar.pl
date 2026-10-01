  $a = $par;
  $gamma = abs(2./$a);
# M. A. Cazalilla Bosonizing one-dimensional cold atomic gases Journal of Physics B AMOP 37, S1 (2004)
# Eqs (79) - (81)
# I checked that with gamma = 8 reproduces the KL very well for all ranges
  if($gamma > 8.) {
    $Kpar = 1 + 4./($gamma);
  }
  else {
    $pi = 3.14159265359;
    $Kpar = $pi/sqrt($gamma)/sqrt(1.-sqrt($gamma)/(2.*$pi));
  }
  $Kpar11 = $Kpar;
